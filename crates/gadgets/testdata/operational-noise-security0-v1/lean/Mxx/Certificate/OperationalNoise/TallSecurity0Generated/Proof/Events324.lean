import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events324

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact82944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩]

theorem exact82944RawTermsValid :
    exact82944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19817⟩⟩) exact82944RawTerms .large 82942 .exactZero (none)

def event82945 : Event := .preFoldPolynomial 82944 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩] .exactZero none

def exact82946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩]

def event82946 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19817⟩⟩) 82945 exact82946RawTerms .large 82942 .exactZero (none)

def event82947 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25223⟩⟩)

def event82948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event82949 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event82950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event82951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event82952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event82953 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event82954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event82955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event82956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 82955

def event82957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 82953

def event82958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 82956 .coefficient) (.value (.predecessor 1 82957 .coefficient)))

def event82959 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event82960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 82959

def event82961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 82951

def event82962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82960 .coefficient, .predecessor 1 82961 .coefficient])

def event82963 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82963

def event82965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 82949

def event82966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82965 .coefficient))

def event82967 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11957⟩⟩) 0 ⟨5536⟩ 82967

def event82969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11957⟩⟩) (.authority (.programFamilyFact))

def exact82970RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact82970RawTermsValid :
    exact82970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82970 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11957⟩⟩) exact82970RawTerms (.finite 36) 82969 .exactZero (none)

def event82971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9715⟩⟩) 0 ⟨5536⟩ 82967

def event82972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9715⟩⟩) (.authority (.programFamilyFact))

def exact82973RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩, (1)⟩]

theorem exact82973RawTermsValid :
    exact82973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9715⟩⟩) exact82973RawTerms (.finite 36) 82972 .exactZero (none)

def event82974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 0 ⟨9715⟩ 82973

def event82975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 1 ⟨11957⟩ 82970

def event82976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.product (.predecessor 0 82974 .coefficient) (.predecessor 1 82975 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82977 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11958⟩⟩, .operator (⟨82973, 0⟩, ⟨82970, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩)

def exact82978RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact82978RawTermsValid :
    exact82978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11958⟩⟩) exact82978RawTerms (.finite 1296) 82976 .exactZero (none)

def event82979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11959⟩⟩) 0 ⟨11958⟩ 82978

def event82980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.identity (.predecessor 0 82979 .coefficient))

def event82981 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.finite 1296)

def event82982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23121⟩⟩) 0 ⟨11959⟩ 82981

def event82983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23121⟩⟩) (.authority (.programFamilyFact))

def event82984 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23121⟩⟩) (.finite 3720)

def event82985 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event82986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23122⟩⟩) 0 ⟨6689⟩ 82985

def event82987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23122⟩⟩) 1 ⟨23121⟩ 82984

def event82988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23122⟩⟩) (.authority (.operator))

def exact82989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩]

theorem exact82989RawTermsValid :
    exact82989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82989 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23122⟩⟩) exact82989RawTerms .large 82988 .exactZero (none)

def event82990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25219⟩⟩) 0 ⟨23122⟩ 82989

def event82991 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25219⟩⟩) (.authority (.operator))

def exact82992RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩]

theorem exact82992RawTermsValid :
    exact82992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82992 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25219⟩⟩) exact82992RawTerms (.finite 8192) 82991 .exactZero (none)

def event82993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event82994 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event82995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12053⟩⟩) 0 ⟨11959⟩ 82981

def event82996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12053⟩⟩) 1 ⟨110⟩ 82994

def event82997 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12053⟩⟩) (.sum [.predecessor 0 82995 .coefficient, .predecessor 1 82996 .coefficient])

def event82998 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12053⟩⟩) (.finite 1296)

def event82999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12054⟩⟩) 0 ⟨12053⟩ 82998

def event83000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12054⟩⟩) (.identity (.predecessor 0 82999 .coefficient))

def exact83001RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact83001RawTermsValid :
    exact83001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12054⟩⟩) exact83001RawTerms (.finite 1296) 83000 .exactZero (none)

def event83002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact83003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83003RawTermsValid :
    exact83003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83003 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact83003RawTerms .large 83002 .exactZero (none)

def event83004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12055⟩⟩) 0 ⟨6544⟩ 83003

def event83005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12055⟩⟩) 1 ⟨12054⟩ 83001

def event83006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12055⟩⟩) (.product (.predecessor 0 83004 .coefficient) (.predecessor 1 83005 .coefficient) (⟨false, false, none, none, none⟩))

def event83007 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12055⟩⟩, .operator (⟨83003, 0⟩, ⟨83001, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83008RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83008RawTermsValid :
    exact83008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83008 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12055⟩⟩) exact83008RawTerms .large 83006 .exactZero (none)

def event83009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 82985

def event83010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact83011RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact83011RawTermsValid :
    exact83011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83011 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact83011RawTerms .large 83010 .exactZero (none)

def event83012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6784⟩⟩) 0 ⟨6757⟩ 83011

def event83013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6784⟩⟩) (.identity (.predecessor 0 83012 .coefficient))

def exact83014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩]

theorem exact83014RawTermsValid :
    exact83014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83014 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6784⟩⟩) exact83014RawTerms .large 83013 .exactZero (none)

def event83015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7864⟩⟩) 0 ⟨6784⟩ 83014

def event83016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7864⟩⟩) (.authority (.operator))

def exact83017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact83017RawTermsValid :
    exact83017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7864⟩⟩) exact83017RawTerms (.finite 8192) 83016 .exactZero (none)

def event83018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7865⟩⟩) 0 ⟨7864⟩ 83017

def event83019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7865⟩⟩) 1 ⟨2348⟩ 82951

def event83020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7865⟩⟩) (.scale (.predecessor 0 83018 .coefficient) (.value (.predecessor 1 83019 .coefficient)))

def exact83021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact83021RawTermsValid :
    exact83021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83021 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7865⟩⟩) exact83021RawTerms (.finite 8192) 83020 .exactZero (none)

def event83022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6764⟩⟩) 0 ⟨6757⟩ 83011

def event83023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6764⟩⟩) (.identity (.predecessor 0 83022 .coefficient))

def exact83024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩]

theorem exact83024RawTermsValid :
    exact83024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6764⟩⟩) exact83024RawTerms .large 83023 .exactZero (none)

def event83025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7866⟩⟩) 0 ⟨6764⟩ 83024

def event83026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7866⟩⟩) 1 ⟨7865⟩ 83021

def event83027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7866⟩⟩) (.product (.predecessor 0 83025 .coefficient) (.predecessor 1 83026 .coefficient) (⟨false, false, none, none, none⟩))

def event83028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7866⟩⟩, .operator (⟨83024, 0⟩, ⟨83021, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩)

def exact83029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact83029RawTermsValid :
    exact83029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7866⟩⟩) exact83029RawTerms .large 83027 .exactZero (none)

def event83030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12056⟩⟩) 0 ⟨7866⟩ 83029

def event83031 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12056⟩⟩) 1 ⟨12055⟩ 83008

def event83032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12056⟩⟩) (.sum [.predecessor 0 83030 .coefficient, .predecessor 1 83031 .coefficient])

def exact83033RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83033RawTermsValid :
    exact83033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83033 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12056⟩⟩) exact83033RawTerms .large 83032 .exactZero (none)

def event83034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25222⟩⟩) 0 ⟨12056⟩ 83033

def event83035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25222⟩⟩) 1 ⟨25219⟩ 82992

def event83036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25222⟩⟩) (.product (.predecessor 0 83034 .coefficient) (.predecessor 1 83035 .coefficient) (⟨false, false, none, none, none⟩))

def event83037 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25222⟩⟩, .operator (⟨83033, 0⟩, ⟨82992, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩)

def event83038 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25222⟩⟩, .operator (⟨83033, 1⟩, ⟨82992, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩)

def event83039 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25222⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25219⟩⟩) ⟨23122⟩ 82989)

def event83040 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25222⟩⟩, .relation 83039 0, ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (-1)⟩)

def exact83041RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (-1)⟩]

theorem exact83041RawTermsValid :
    exact83041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83041 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25222⟩⟩) exact83041RawTerms .large 83036 .exactZero (none)

def event83042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16381⟩⟩) 0 ⟨11959⟩ 82981

def event83043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16381⟩⟩) (.authority (.programFamilyFact))

def exact83044RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], []⟩, (1)⟩]

theorem exact83044RawTermsValid :
    exact83044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16381⟩⟩) exact83044RawTerms (.finite 36) 83043 .exactZero (none)

def event83045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16383⟩⟩) 0 ⟨6544⟩ 83003

def event83046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16383⟩⟩) 1 ⟨16381⟩ 83044

def event83047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16383⟩⟩) (.product (.predecessor 0 83045 .coefficient) (.predecessor 1 83046 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16383⟩⟩, .operator (⟨83003, 0⟩, ⟨83044, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83049RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83049RawTermsValid :
    exact83049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16383⟩⟩) exact83049RawTerms .large 83047 .exactZero (none)

def event83050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 82985

def event83051 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact83052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact83052RawTermsValid :
    exact83052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact83052RawTerms .large 83051 .exactZero (none)

def event83053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16384⟩⟩) 0 ⟨6701⟩ 83052

def event83054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16384⟩⟩) 1 ⟨16383⟩ 83049

def event83055 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16384⟩⟩) (.sum [.predecessor 0 83053 .coefficient, .predecessor 1 83054 .coefficient])

def exact83056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83056RawTermsValid :
    exact83056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83056 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16384⟩⟩) exact83056RawTerms .large 83055 .exactZero (none)

def event83057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25223⟩⟩) 0 ⟨16384⟩ 83056

def event83058 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25223⟩⟩) 1 ⟨25222⟩ 83041

def event83059 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25223⟩⟩) (.sum [.predecessor 0 83057 .coefficient, .predecessor 1 83058 .coefficient])

def exact83060RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83060RawTermsValid :
    exact83060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83060 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25223⟩⟩) exact83060RawTerms .large 83059 .exactZero (none)

def event83061 : Event := .preFoldPolynomial 83060 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83062RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event83062 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25223⟩⟩) 83061 exact83062RawTerms .large 83059 .exactZero (none)

def event83063 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11959⟩⟩) ⟨⟨114⟩, ⟨19⟩, ⟨109⟩⟩ ⟨82899, 83063⟩

def event83064 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19819⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩) (1) 0 2 (.universal 83063 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩) (none) 83062)

def event83065 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19819⟩⟩, .relation 83064 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩)

def event83066 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19819⟩⟩, .relation 83064 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩)

def event83067 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19819⟩⟩, .relation 83064 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩)

def event83068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19819⟩⟩, .relation 83064 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact83069RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83069RawTermsValid :
    exact83069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19819⟩⟩) exact83069RawTerms .large 82895 (.finite 1811303510016) (some (82897))

def event83070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25221⟩⟩) 0 ⟨19819⟩ 83069

def event83071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25221⟩⟩) 1 ⟨25220⟩ 82885

def event83072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25221⟩⟩) (.sum [.predecessor 0 83070 .coefficient, .predecessor 1 83071 .coefficient])

def event83073 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25221⟩⟩, .operator (⟨83069, 2⟩, ⟨82885, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (-1)⟩)

def event83074 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25221⟩⟩, .operator (⟨83069, 1⟩, ⟨82885, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩)

def event83075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25221⟩⟩) (.sum [.result 83069 .summary, .result 82885 .summary])

def exact83076RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83076RawTermsValid :
    exact83076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25221⟩⟩) exact83076RawTerms .large 83072 (.finite 352115681275904) (some (83075))

def event83077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28736⟩⟩) 0 ⟨25221⟩ 83076

def event83078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28736⟩⟩) 1 ⟨28734⟩ 82801

def event83079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28736⟩⟩) (.product (.predecessor 0 83077 .coefficient) (.predecessor 1 83078 .coefficient) (⟨false, false, none, none, none⟩))

def event83080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28736⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩) [⟨.result 82801 .coefficient, false, none⟩])

def event83081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28736⟩⟩) (.product (.result 83076 .summary) (.transfer 83080) (⟨false, false, none, none, none⟩))

def event83082 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28736⟩⟩, .operator (⟨83076, 0⟩, ⟨82801, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩)

def event83083 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28736⟩⟩, .operator (⟨83076, 1⟩, ⟨82801, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩)

def event83084 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28736⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28734⟩⟩) ⟨24414⟩ 82798)

def event83085 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28736⟩⟩, .relation 83084 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (-1)⟩)

def exact83086RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (-1)⟩]

theorem exact83086RawTermsValid :
    exact83086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83086 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28736⟩⟩) exact83086RawTerms .large 83079 (.finite 1292270184133468094464) (some (83081))

def event83087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21976⟩⟩) 0 ⟨16382⟩ 3983

def event83088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21976⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact83089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩]

theorem exact83089RawTermsValid :
    exact83089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21976⟩⟩) exact83089RawTerms (.finite 136065468) 83088 .exactZero (none)

def event83090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21978⟩⟩) 0 ⟨21976⟩ 83089

def event83091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21978⟩⟩) 1 ⟨2348⟩ 4

def event83092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21978⟩⟩) (.scale (.predecessor 0 83090 .coefficient) (.value (.predecessor 1 83091 .coefficient)))

def exact83093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩]

theorem exact83093RawTermsValid :
    exact83093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83093 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21978⟩⟩) exact83093RawTerms (.finite 136065468) 83092 .exactZero (none)

def event83094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21979⟩⟩) 0 ⟨5541⟩ 80012

def event83095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21979⟩⟩) 1 ⟨21978⟩ 83093

def event83096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21979⟩⟩) (.product (.predecessor 0 83094 .coefficient) (.predecessor 1 83095 .coefficient) (⟨false, false, none, none, none⟩))

def event83097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21979⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩) [⟨.result 83089 .coefficient, false, none⟩])

def event83098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21979⟩⟩) (.product (.result 80012 .summary) (.transfer 83097) (⟨false, false, none, none, none⟩))

def event83099 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21979⟩⟩, .operator (⟨80012, 0⟩, ⟨83093, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩)

def event83100 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21977⟩⟩)

def event83101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83102 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83104 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83106 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83108 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83109 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83108

def event83110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83106

def event83111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83109 .coefficient) (.value (.predecessor 1 83110 .coefficient)))

def event83112 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83112

def event83114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83104

def event83115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83113 .coefficient, .predecessor 1 83114 .coefficient])

def event83116 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83116

def event83118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83102

def event83119 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83118 .coefficient))

def event83120 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11957⟩⟩) 0 ⟨5536⟩ 83120

def event83122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11957⟩⟩) (.authority (.programFamilyFact))

def exact83123RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact83123RawTermsValid :
    exact83123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83123 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11957⟩⟩) exact83123RawTerms (.finite 36) 83122 .exactZero (none)

def event83124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9715⟩⟩) 0 ⟨5536⟩ 83120

def event83125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9715⟩⟩) (.authority (.programFamilyFact))

def exact83126RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩, (1)⟩]

theorem exact83126RawTermsValid :
    exact83126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9715⟩⟩) exact83126RawTerms (.finite 36) 83125 .exactZero (none)

def event83127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 0 ⟨9715⟩ 83126

def event83128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 1 ⟨11957⟩ 83123

def event83129 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.product (.predecessor 0 83127 .coefficient) (.predecessor 1 83128 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩) [⟨.result 83126 .coefficient, true, some 1⟩, ⟨.result 83123 .coefficient, true, some 1⟩])

def event83131 : Event := .survivorFold (1) 83130

def exact83132RawTerms : List Term := []

theorem exact83132RawTermsValid :
    exact83132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11958⟩⟩) exact83132RawTerms (.finite 1296) 83129 (.finite 1296) (some (83130))

def event83133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11959⟩⟩) 0 ⟨11958⟩ 83132

def event83134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.identity (.predecessor 0 83133 .coefficient))

def event83135 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.finite 1296)

def event83136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16381⟩⟩) 0 ⟨11959⟩ 83135

def event83137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16381⟩⟩) (.authority (.programFamilyFact))

def exact83138RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], []⟩, (1)⟩]

theorem exact83138RawTermsValid :
    exact83138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16381⟩⟩) exact83138RawTerms (.finite 36) 83137 .exactZero (none)

def event83139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16382⟩⟩) 0 ⟨16381⟩ 83138

def event83140 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16382⟩⟩) (.identity (.predecessor 0 83139 .coefficient))

def event83141 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16382⟩⟩) (.finite 36)

def event83142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21976⟩⟩) 0 ⟨16382⟩ 83141

def event83143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21976⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact83144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩]

theorem exact83144RawTermsValid :
    exact83144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21976⟩⟩) exact83144RawTerms (.finite 136065468) 83143 .exactZero (none)

def event83145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact83146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact83146RawTermsValid :
    exact83146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact83146RawTerms .large 83145 .exactZero (none)

def event83147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21977⟩⟩) 0 ⟨6⟩ 83146

def event83148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21977⟩⟩) 1 ⟨21976⟩ 83144

def event83149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21977⟩⟩) (.product (.predecessor 0 83147 .coefficient) (.predecessor 1 83148 .coefficient) (⟨false, false, none, none, none⟩))

def event83150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21977⟩⟩, .operator (⟨83146, 0⟩, ⟨83144, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩)

def exact83151RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩]

theorem exact83151RawTermsValid :
    exact83151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21977⟩⟩) exact83151RawTerms .large 83149 .exactZero (none)

def event83152 : Event := .preFoldPolynomial 83151 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩] .exactZero none

def exact83153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩, (1)⟩]

def event83153 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21977⟩⟩) 83152 exact83153RawTerms .large 83149 .exactZero (none)

def event83154 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28739⟩⟩)

def event83155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83156 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83158 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83160 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83162 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83162

def event83164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83160

def event83165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83163 .coefficient) (.value (.predecessor 1 83164 .coefficient)))

def event83166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83166

def event83168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83158

def event83169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83167 .coefficient, .predecessor 1 83168 .coefficient])

def event83170 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83170

def event83172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83156

def event83173 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83172 .coefficient))

def event83174 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11957⟩⟩) 0 ⟨5536⟩ 83174

def event83176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11957⟩⟩) (.authority (.programFamilyFact))

def exact83177RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact83177RawTermsValid :
    exact83177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83177 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11957⟩⟩) exact83177RawTerms (.finite 36) 83176 .exactZero (none)

def event83178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9715⟩⟩) 0 ⟨5536⟩ 83174

def event83179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9715⟩⟩) (.authority (.programFamilyFact))

def exact83180RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩, (1)⟩]

theorem exact83180RawTermsValid :
    exact83180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83180 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9715⟩⟩) exact83180RawTerms (.finite 36) 83179 .exactZero (none)

def event83181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 0 ⟨9715⟩ 83180

def event83182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 1 ⟨11957⟩ 83177

def event83183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.product (.predecessor 0 83181 .coefficient) (.predecessor 1 83182 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83184 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11958⟩⟩, .operator (⟨83180, 0⟩, ⟨83177, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩)

def exact83185RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact83185RawTermsValid :
    exact83185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11958⟩⟩) exact83185RawTerms (.finite 1296) 83183 .exactZero (none)

def event83186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11959⟩⟩) 0 ⟨11958⟩ 83185

def event83187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.identity (.predecessor 0 83186 .coefficient))

def event83188 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.finite 1296)

def event83189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16381⟩⟩) 0 ⟨11959⟩ 83188

def event83190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16381⟩⟩) (.authority (.programFamilyFact))

def exact83191RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], []⟩, (1)⟩]

theorem exact83191RawTermsValid :
    exact83191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16381⟩⟩) exact83191RawTerms (.finite 36) 83190 .exactZero (none)

def event83192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16382⟩⟩) 0 ⟨16381⟩ 83191

def event83193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16382⟩⟩) (.identity (.predecessor 0 83192 .coefficient))

def event83194 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16382⟩⟩) (.finite 36)

def event83195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24412⟩⟩) 0 ⟨16382⟩ 83194

def event83196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24412⟩⟩) (.authority (.programFamilyFact))

def event83197 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24412⟩⟩) (.finite 3720)

def event83198 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event83199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24414⟩⟩) 0 ⟨6689⟩ 83198

def eventLeaf5184 : Array AnnotatedEvent := #[
  { event := event82944
    frameStart := 82899 },
  { event := event82945
    frameStart := 82899 },
  { event := event82946
    frameStart := 82899 },
  { event := event82947
    frameStart := 82947 },
  { event := event82948
    frameStart := 82947 },
  { event := event82949
    frameStart := 82947 },
  { event := event82950
    frameStart := 82947 },
  { event := event82951
    frameStart := 82947 },
  { event := event82952
    frameStart := 82947 },
  { event := event82953
    frameStart := 82947 },
  { event := event82954
    frameStart := 82947 },
  { event := event82955
    frameStart := 82947 },
  { event := event82956
    frameStart := 82947 },
  { event := event82957
    frameStart := 82947 },
  { event := event82958
    frameStart := 82947 },
  { event := event82959
    frameStart := 82947 }
]

def eventLeaf5185 : Array AnnotatedEvent := #[
  { event := event82960
    frameStart := 82947 },
  { event := event82961
    frameStart := 82947 },
  { event := event82962
    frameStart := 82947 },
  { event := event82963
    frameStart := 82947 },
  { event := event82964
    frameStart := 82947 },
  { event := event82965
    frameStart := 82947 },
  { event := event82966
    frameStart := 82947 },
  { event := event82967
    frameStart := 82947 },
  { event := event82968
    frameStart := 82947 },
  { event := event82969
    frameStart := 82947 },
  { event := event82970
    frameStart := 82947 },
  { event := event82971
    frameStart := 82947 },
  { event := event82972
    frameStart := 82947 },
  { event := event82973
    frameStart := 82947 },
  { event := event82974
    frameStart := 82947 },
  { event := event82975
    frameStart := 82947 }
]

def eventLeaf5186 : Array AnnotatedEvent := #[
  { event := event82976
    frameStart := 82947 },
  { event := event82977
    frameStart := 82947 },
  { event := event82978
    frameStart := 82947 },
  { event := event82979
    frameStart := 82947 },
  { event := event82980
    frameStart := 82947 },
  { event := event82981
    frameStart := 82947 },
  { event := event82982
    frameStart := 82947 },
  { event := event82983
    frameStart := 82947 },
  { event := event82984
    frameStart := 82947 },
  { event := event82985
    frameStart := 82947 },
  { event := event82986
    frameStart := 82947 },
  { event := event82987
    frameStart := 82947 },
  { event := event82988
    frameStart := 82947 },
  { event := event82989
    frameStart := 82947 },
  { event := event82990
    frameStart := 82947 },
  { event := event82991
    frameStart := 82947 }
]

def eventLeaf5187 : Array AnnotatedEvent := #[
  { event := event82992
    frameStart := 82947 },
  { event := event82993
    frameStart := 82947 },
  { event := event82994
    frameStart := 82947 },
  { event := event82995
    frameStart := 82947 },
  { event := event82996
    frameStart := 82947 },
  { event := event82997
    frameStart := 82947 },
  { event := event82998
    frameStart := 82947 },
  { event := event82999
    frameStart := 82947 },
  { event := event83000
    frameStart := 82947 },
  { event := event83001
    frameStart := 82947 },
  { event := event83002
    frameStart := 82947 },
  { event := event83003
    frameStart := 82947 },
  { event := event83004
    frameStart := 82947 },
  { event := event83005
    frameStart := 82947 },
  { event := event83006
    frameStart := 82947 },
  { event := event83007
    frameStart := 82947 }
]

def eventLeaf5188 : Array AnnotatedEvent := #[
  { event := event83008
    frameStart := 82947 },
  { event := event83009
    frameStart := 82947 },
  { event := event83010
    frameStart := 82947 },
  { event := event83011
    frameStart := 82947 },
  { event := event83012
    frameStart := 82947 },
  { event := event83013
    frameStart := 82947 },
  { event := event83014
    frameStart := 82947 },
  { event := event83015
    frameStart := 82947 },
  { event := event83016
    frameStart := 82947 },
  { event := event83017
    frameStart := 82947 },
  { event := event83018
    frameStart := 82947 },
  { event := event83019
    frameStart := 82947 },
  { event := event83020
    frameStart := 82947 },
  { event := event83021
    frameStart := 82947 },
  { event := event83022
    frameStart := 82947 },
  { event := event83023
    frameStart := 82947 }
]

def eventLeaf5189 : Array AnnotatedEvent := #[
  { event := event83024
    frameStart := 82947 },
  { event := event83025
    frameStart := 82947 },
  { event := event83026
    frameStart := 82947 },
  { event := event83027
    frameStart := 82947 },
  { event := event83028
    frameStart := 82947 },
  { event := event83029
    frameStart := 82947 },
  { event := event83030
    frameStart := 82947 },
  { event := event83031
    frameStart := 82947 },
  { event := event83032
    frameStart := 82947 },
  { event := event83033
    frameStart := 82947 },
  { event := event83034
    frameStart := 82947 },
  { event := event83035
    frameStart := 82947 },
  { event := event83036
    frameStart := 82947 },
  { event := event83037
    frameStart := 82947 },
  { event := event83038
    frameStart := 82947 },
  { event := event83039
    frameStart := 82947 }
]

def eventLeaf5190 : Array AnnotatedEvent := #[
  { event := event83040
    frameStart := 82947 },
  { event := event83041
    frameStart := 82947 },
  { event := event83042
    frameStart := 82947 },
  { event := event83043
    frameStart := 82947 },
  { event := event83044
    frameStart := 82947 },
  { event := event83045
    frameStart := 82947 },
  { event := event83046
    frameStart := 82947 },
  { event := event83047
    frameStart := 82947 },
  { event := event83048
    frameStart := 82947 },
  { event := event83049
    frameStart := 82947 },
  { event := event83050
    frameStart := 82947 },
  { event := event83051
    frameStart := 82947 },
  { event := event83052
    frameStart := 82947 },
  { event := event83053
    frameStart := 82947 },
  { event := event83054
    frameStart := 82947 },
  { event := event83055
    frameStart := 82947 }
]

def eventLeaf5191 : Array AnnotatedEvent := #[
  { event := event83056
    frameStart := 82947 },
  { event := event83057
    frameStart := 82947 },
  { event := event83058
    frameStart := 82947 },
  { event := event83059
    frameStart := 82947 },
  { event := event83060
    frameStart := 82947 },
  { event := event83061
    frameStart := 82947 },
  { event := event83062
    frameStart := 82947 },
  { event := event83063
    frameStart := 0 },
  { event := event83064
    frameStart := 0 },
  { event := event83065
    frameStart := 0 },
  { event := event83066
    frameStart := 0 },
  { event := event83067
    frameStart := 0 },
  { event := event83068
    frameStart := 0 },
  { event := event83069
    frameStart := 0 },
  { event := event83070
    frameStart := 0 },
  { event := event83071
    frameStart := 0 }
]

def eventLeaf5192 : Array AnnotatedEvent := #[
  { event := event83072
    frameStart := 0 },
  { event := event83073
    frameStart := 0 },
  { event := event83074
    frameStart := 0 },
  { event := event83075
    frameStart := 0 },
  { event := event83076
    frameStart := 0 },
  { event := event83077
    frameStart := 0 },
  { event := event83078
    frameStart := 0 },
  { event := event83079
    frameStart := 0 },
  { event := event83080
    frameStart := 0 },
  { event := event83081
    frameStart := 0 },
  { event := event83082
    frameStart := 0 },
  { event := event83083
    frameStart := 0 },
  { event := event83084
    frameStart := 0 },
  { event := event83085
    frameStart := 0 },
  { event := event83086
    frameStart := 0 },
  { event := event83087
    frameStart := 0 }
]

def eventLeaf5193 : Array AnnotatedEvent := #[
  { event := event83088
    frameStart := 0 },
  { event := event83089
    frameStart := 0 },
  { event := event83090
    frameStart := 0 },
  { event := event83091
    frameStart := 0 },
  { event := event83092
    frameStart := 0 },
  { event := event83093
    frameStart := 0 },
  { event := event83094
    frameStart := 0 },
  { event := event83095
    frameStart := 0 },
  { event := event83096
    frameStart := 0 },
  { event := event83097
    frameStart := 0 },
  { event := event83098
    frameStart := 0 },
  { event := event83099
    frameStart := 0 },
  { event := event83100
    frameStart := 83100 },
  { event := event83101
    frameStart := 83100 },
  { event := event83102
    frameStart := 83100 },
  { event := event83103
    frameStart := 83100 }
]

def eventLeaf5194 : Array AnnotatedEvent := #[
  { event := event83104
    frameStart := 83100 },
  { event := event83105
    frameStart := 83100 },
  { event := event83106
    frameStart := 83100 },
  { event := event83107
    frameStart := 83100 },
  { event := event83108
    frameStart := 83100 },
  { event := event83109
    frameStart := 83100 },
  { event := event83110
    frameStart := 83100 },
  { event := event83111
    frameStart := 83100 },
  { event := event83112
    frameStart := 83100 },
  { event := event83113
    frameStart := 83100 },
  { event := event83114
    frameStart := 83100 },
  { event := event83115
    frameStart := 83100 },
  { event := event83116
    frameStart := 83100 },
  { event := event83117
    frameStart := 83100 },
  { event := event83118
    frameStart := 83100 },
  { event := event83119
    frameStart := 83100 }
]

def eventLeaf5195 : Array AnnotatedEvent := #[
  { event := event83120
    frameStart := 83100 },
  { event := event83121
    frameStart := 83100 },
  { event := event83122
    frameStart := 83100 },
  { event := event83123
    frameStart := 83100 },
  { event := event83124
    frameStart := 83100 },
  { event := event83125
    frameStart := 83100 },
  { event := event83126
    frameStart := 83100 },
  { event := event83127
    frameStart := 83100 },
  { event := event83128
    frameStart := 83100 },
  { event := event83129
    frameStart := 83100 },
  { event := event83130
    frameStart := 83100 },
  { event := event83131
    frameStart := 83100 },
  { event := event83132
    frameStart := 83100 },
  { event := event83133
    frameStart := 83100 },
  { event := event83134
    frameStart := 83100 },
  { event := event83135
    frameStart := 83100 }
]

def eventLeaf5196 : Array AnnotatedEvent := #[
  { event := event83136
    frameStart := 83100 },
  { event := event83137
    frameStart := 83100 },
  { event := event83138
    frameStart := 83100 },
  { event := event83139
    frameStart := 83100 },
  { event := event83140
    frameStart := 83100 },
  { event := event83141
    frameStart := 83100 },
  { event := event83142
    frameStart := 83100 },
  { event := event83143
    frameStart := 83100 },
  { event := event83144
    frameStart := 83100 },
  { event := event83145
    frameStart := 83100 },
  { event := event83146
    frameStart := 83100 },
  { event := event83147
    frameStart := 83100 },
  { event := event83148
    frameStart := 83100 },
  { event := event83149
    frameStart := 83100 },
  { event := event83150
    frameStart := 83100 },
  { event := event83151
    frameStart := 83100 }
]

def eventLeaf5197 : Array AnnotatedEvent := #[
  { event := event83152
    frameStart := 83100 },
  { event := event83153
    frameStart := 83100 },
  { event := event83154
    frameStart := 83154 },
  { event := event83155
    frameStart := 83154 },
  { event := event83156
    frameStart := 83154 },
  { event := event83157
    frameStart := 83154 },
  { event := event83158
    frameStart := 83154 },
  { event := event83159
    frameStart := 83154 },
  { event := event83160
    frameStart := 83154 },
  { event := event83161
    frameStart := 83154 },
  { event := event83162
    frameStart := 83154 },
  { event := event83163
    frameStart := 83154 },
  { event := event83164
    frameStart := 83154 },
  { event := event83165
    frameStart := 83154 },
  { event := event83166
    frameStart := 83154 },
  { event := event83167
    frameStart := 83154 }
]

def eventLeaf5198 : Array AnnotatedEvent := #[
  { event := event83168
    frameStart := 83154 },
  { event := event83169
    frameStart := 83154 },
  { event := event83170
    frameStart := 83154 },
  { event := event83171
    frameStart := 83154 },
  { event := event83172
    frameStart := 83154 },
  { event := event83173
    frameStart := 83154 },
  { event := event83174
    frameStart := 83154 },
  { event := event83175
    frameStart := 83154 },
  { event := event83176
    frameStart := 83154 },
  { event := event83177
    frameStart := 83154 },
  { event := event83178
    frameStart := 83154 },
  { event := event83179
    frameStart := 83154 },
  { event := event83180
    frameStart := 83154 },
  { event := event83181
    frameStart := 83154 },
  { event := event83182
    frameStart := 83154 },
  { event := event83183
    frameStart := 83154 }
]

def eventLeaf5199 : Array AnnotatedEvent := #[
  { event := event83184
    frameStart := 83154 },
  { event := event83185
    frameStart := 83154 },
  { event := event83186
    frameStart := 83154 },
  { event := event83187
    frameStart := 83154 },
  { event := event83188
    frameStart := 83154 },
  { event := event83189
    frameStart := 83154 },
  { event := event83190
    frameStart := 83154 },
  { event := event83191
    frameStart := 83154 },
  { event := event83192
    frameStart := 83154 },
  { event := event83193
    frameStart := 83154 },
  { event := event83194
    frameStart := 83154 },
  { event := event83195
    frameStart := 83154 },
  { event := event83196
    frameStart := 83154 },
  { event := event83197
    frameStart := 83154 },
  { event := event83198
    frameStart := 83154 },
  { event := event83199
    frameStart := 83154 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events324
