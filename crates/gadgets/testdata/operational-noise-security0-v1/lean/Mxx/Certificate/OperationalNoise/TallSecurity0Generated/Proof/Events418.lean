import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events418

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event107008 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event107009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event107010 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event107011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 107010

def event107012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 107008

def event107013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 107011 .coefficient) (.value (.predecessor 1 107012 .coefficient)))

def event107014 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event107015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10456⟩⟩) 0 ⟨5503⟩ 107014

def event107016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10456⟩⟩) (.authority (.programFamilyFact))

def exact107017RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩, (1)⟩]

theorem exact107017RawTermsValid :
    exact107017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10456⟩⟩) exact107017RawTerms (.finite 2) 107016 .exactZero (none)

def event107018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9385⟩⟩) 0 ⟨5503⟩ 107014

def event107019 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9385⟩⟩) (.authority (.programFamilyFact))

def exact107020RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩], []⟩, (1)⟩]

theorem exact107020RawTermsValid :
    exact107020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9385⟩⟩) exact107020RawTerms (.finite 2) 107019 .exactZero (none)

def event107021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10457⟩⟩) 0 ⟨9385⟩ 107020

def event107022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10457⟩⟩) 1 ⟨10456⟩ 107017

def event107023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10457⟩⟩) (.product (.predecessor 0 107021 .coefficient) (.predecessor 1 107022 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event107024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10457⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩) [⟨.result 107020 .coefficient, true, some 1⟩, ⟨.result 107017 .coefficient, true, some 1⟩])

def event107025 : Event := .survivorFold (1) 107024

def exact107026RawTerms : List Term := []

theorem exact107026RawTermsValid :
    exact107026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107026 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10457⟩⟩) exact107026RawTerms (.finite 4) 107023 (.finite 4) (some (107024))

def event107027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10458⟩⟩) 0 ⟨10457⟩ 107026

def event107028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10458⟩⟩) (.identity (.predecessor 0 107027 .coefficient))

def event107029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10458⟩⟩) (.finite 4)

def event107030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14782⟩⟩) 0 ⟨10458⟩ 107029

def event107031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14782⟩⟩) (.authority (.programFamilyFact))

def exact107032RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], []⟩, (1)⟩]

theorem exact107032RawTermsValid :
    exact107032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14782⟩⟩) exact107032RawTerms (.finite 2) 107031 .exactZero (none)

def event107033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14783⟩⟩) 0 ⟨14782⟩ 107032

def event107034 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14783⟩⟩) (.identity (.predecessor 0 107033 .coefficient))

def event107035 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14783⟩⟩) (.finite 2)

def event107036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20309⟩⟩) 0 ⟨14783⟩ 107035

def event107037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20309⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact107038RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩]

theorem exact107038RawTermsValid :
    exact107038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20309⟩⟩) exact107038RawTerms (.finite 136065468) 107037 .exactZero (none)

def event107039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact107040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact107040RawTermsValid :
    exact107040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107040 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact107040RawTerms .large 107039 .exactZero (none)

def event107041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20310⟩⟩) 0 ⟨6⟩ 107040

def event107042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20310⟩⟩) 1 ⟨20309⟩ 107038

def event107043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20310⟩⟩) (.product (.predecessor 0 107041 .coefficient) (.predecessor 1 107042 .coefficient) (⟨false, false, none, none, none⟩))

def event107044 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20310⟩⟩, .operator (⟨107040, 0⟩, ⟨107038, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩)

def exact107045RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩]

theorem exact107045RawTermsValid :
    exact107045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107045 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20310⟩⟩) exact107045RawTerms .large 107043 .exactZero (none)

def event107046 : Event := .preFoldPolynomial 107045 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩] .exactZero none

def exact107047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩, (1)⟩]

def event107047 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20310⟩⟩) 107046 exact107047RawTerms .large 107043 .exactZero (none)

def event107048 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26325⟩⟩)

def event107049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event107050 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event107051 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event107052 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event107053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 107052

def event107054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 107050

def event107055 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 107053 .coefficient) (.value (.predecessor 1 107054 .coefficient)))

def event107056 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event107057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10456⟩⟩) 0 ⟨5503⟩ 107056

def event107058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10456⟩⟩) (.authority (.programFamilyFact))

def exact107059RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩, (1)⟩]

theorem exact107059RawTermsValid :
    exact107059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10456⟩⟩) exact107059RawTerms (.finite 2) 107058 .exactZero (none)

def event107060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9385⟩⟩) 0 ⟨5503⟩ 107056

def event107061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9385⟩⟩) (.authority (.programFamilyFact))

def exact107062RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩], []⟩, (1)⟩]

theorem exact107062RawTermsValid :
    exact107062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107062 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9385⟩⟩) exact107062RawTerms (.finite 2) 107061 .exactZero (none)

def event107063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10457⟩⟩) 0 ⟨9385⟩ 107062

def event107064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10457⟩⟩) 1 ⟨10456⟩ 107059

def event107065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10457⟩⟩) (.product (.predecessor 0 107063 .coefficient) (.predecessor 1 107064 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event107066 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10457⟩⟩, .operator (⟨107062, 0⟩, ⟨107059, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩, (1)⟩)

def exact107067RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], []⟩, (1)⟩]

theorem exact107067RawTermsValid :
    exact107067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107067 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10457⟩⟩) exact107067RawTerms (.finite 4) 107065 .exactZero (none)

def event107068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10458⟩⟩) 0 ⟨10457⟩ 107067

def event107069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10458⟩⟩) (.identity (.predecessor 0 107068 .coefficient))

def event107070 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10458⟩⟩) (.finite 4)

def event107071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14782⟩⟩) 0 ⟨10458⟩ 107070

def event107072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14782⟩⟩) (.authority (.programFamilyFact))

def exact107073RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], []⟩, (1)⟩]

theorem exact107073RawTermsValid :
    exact107073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107073 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14782⟩⟩) exact107073RawTerms (.finite 2) 107072 .exactZero (none)

def event107074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14783⟩⟩) 0 ⟨14782⟩ 107073

def event107075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14783⟩⟩) (.identity (.predecessor 0 107074 .coefficient))

def event107076 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14783⟩⟩) (.finite 2)

def event107077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23710⟩⟩) 0 ⟨14783⟩ 107076

def event107078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23710⟩⟩) (.authority (.programFamilyFact))

def event107079 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23710⟩⟩) (.finite 3720)

def event107080 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event107081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23711⟩⟩) 0 ⟨6689⟩ 107080

def event107082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23711⟩⟩) 1 ⟨23710⟩ 107079

def event107083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23711⟩⟩) (.authority (.operator))

def exact107084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩]

theorem exact107084RawTermsValid :
    exact107084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107084 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23711⟩⟩) exact107084RawTerms .large 107083 .exactZero (none)

def event107085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26319⟩⟩) 0 ⟨23711⟩ 107084

def event107086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26319⟩⟩) (.authority (.operator))

def exact107087RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩]

theorem exact107087RawTermsValid :
    exact107087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107087 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26319⟩⟩) exact107087RawTerms (.finite 8192) 107086 .exactZero (none)

def event107088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event107089 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event107090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14824⟩⟩) 0 ⟨14783⟩ 107076

def event107091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14824⟩⟩) 1 ⟨110⟩ 107089

def event107092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14824⟩⟩) (.sum [.predecessor 0 107090 .coefficient, .predecessor 1 107091 .coefficient])

def event107093 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14824⟩⟩) (.finite 2)

def event107094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14825⟩⟩) 0 ⟨14824⟩ 107093

def event107095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14825⟩⟩) (.identity (.predecessor 0 107094 .coefficient))

def exact107096RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], []⟩, (1)⟩]

theorem exact107096RawTermsValid :
    exact107096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14825⟩⟩) exact107096RawTerms (.finite 2) 107095 .exactZero (none)

def event107097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact107098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact107098RawTermsValid :
    exact107098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact107098RawTerms .large 107097 .exactZero (none)

def event107099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14826⟩⟩) 0 ⟨6544⟩ 107098

def event107100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14826⟩⟩) 1 ⟨14825⟩ 107096

def event107101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14826⟩⟩) (.product (.predecessor 0 107099 .coefficient) (.predecessor 1 107100 .coefficient) (⟨false, false, none, none, none⟩))

def event107102 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14826⟩⟩, .operator (⟨107098, 0⟩, ⟨107096, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact107103RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact107103RawTermsValid :
    exact107103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107103 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14826⟩⟩) exact107103RawTerms .large 107101 .exactZero (none)

def event107104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 107080

def event107105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact107106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact107106RawTermsValid :
    exact107106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact107106RawTerms .large 107105 .exactZero (none)

def event107107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14827⟩⟩) 0 ⟨6690⟩ 107106

def event107108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14827⟩⟩) 1 ⟨14826⟩ 107103

def event107109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14827⟩⟩) (.sum [.predecessor 0 107107 .coefficient, .predecessor 1 107108 .coefficient])

def exact107110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107110RawTermsValid :
    exact107110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14827⟩⟩) exact107110RawTerms .large 107109 .exactZero (none)

def event107111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26320⟩⟩) 0 ⟨14827⟩ 107110

def event107112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26320⟩⟩) 1 ⟨26319⟩ 107087

def event107113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26320⟩⟩) (.product (.predecessor 0 107111 .coefficient) (.predecessor 1 107112 .coefficient) (⟨false, false, none, none, none⟩))

def event107114 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26320⟩⟩, .operator (⟨107110, 0⟩, ⟨107087, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩)

def event107115 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26320⟩⟩, .operator (⟨107110, 1⟩, ⟨107087, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩)

def event107116 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26320⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26319⟩⟩) ⟨23711⟩ 107084)

def event107117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26320⟩⟩, .relation 107116 0, ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (-1)⟩)

def exact107118RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (-1)⟩]

theorem exact107118RawTermsValid :
    exact107118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107118 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26320⟩⟩) exact107118RawTerms .large 107113 .exactZero (none)

def event107119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14873⟩⟩) 0 ⟨14783⟩ 107076

def event107120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14873⟩⟩) (.authority (.programFamilyFact))

def exact107121RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], []⟩, (1)⟩]

theorem exact107121RawTermsValid :
    exact107121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107121 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14873⟩⟩) exact107121RawTerms (.finite 2) 107120 .exactZero (none)

def event107122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14876⟩⟩) 0 ⟨6544⟩ 107098

def event107123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14876⟩⟩) 1 ⟨14873⟩ 107121

def event107124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14876⟩⟩) (.product (.predecessor 0 107122 .coefficient) (.predecessor 1 107123 .coefficient) (⟨false, true, none, none, some 1⟩))

def event107125 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14876⟩⟩, .operator (⟨107098, 0⟩, ⟨107121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact107126RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact107126RawTermsValid :
    exact107126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14876⟩⟩) exact107126RawTerms .large 107124 .exactZero (none)

def event107127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6708⟩⟩) 0 ⟨6689⟩ 107080

def event107128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6708⟩⟩) (.authority (.operator))

def exact107129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩]

theorem exact107129RawTermsValid :
    exact107129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6708⟩⟩) exact107129RawTerms .large 107128 .exactZero (none)

def event107130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14877⟩⟩) 0 ⟨6708⟩ 107129

def event107131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14877⟩⟩) 1 ⟨14876⟩ 107126

def event107132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14877⟩⟩) (.sum [.predecessor 0 107130 .coefficient, .predecessor 1 107131 .coefficient])

def exact107133RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107133RawTermsValid :
    exact107133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107133 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14877⟩⟩) exact107133RawTerms .large 107132 .exactZero (none)

def event107134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26325⟩⟩) 0 ⟨14877⟩ 107133

def event107135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26325⟩⟩) 1 ⟨26320⟩ 107118

def event107136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26325⟩⟩) (.sum [.predecessor 0 107134 .coefficient, .predecessor 1 107135 .coefficient])

def exact107137RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107137RawTermsValid :
    exact107137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107137 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26325⟩⟩) exact107137RawTerms .large 107136 .exactZero (none)

def event107138 : Event := .preFoldPolynomial 107137 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact107139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event107139 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26325⟩⟩) 107138 exact107139RawTerms .large 107136 .exactZero (none)

def event107140 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14783⟩⟩) ⟨⟨121⟩, ⟨27⟩, ⟨109⟩⟩ ⟨107006, 107140⟩

def event107141 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20312⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩) (1) 0 2 (.universal 107140 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩) (none) 107139)

def event107142 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20312⟩⟩, .relation 107141 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩)

def event107143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20312⟩⟩, .relation 107141 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩)

def event107144 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20312⟩⟩, .relation 107141 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩)

def event107145 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20312⟩⟩, .relation 107141 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact107146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107146RawTermsValid :
    exact107146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20312⟩⟩) exact107146RawTerms .large 107002 (.finite 1811303510016) (some (107004))

def event107147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26322⟩⟩) 0 ⟨20312⟩ 107146

def event107148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26322⟩⟩) 1 ⟨26321⟩ 106992

def event107149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26322⟩⟩) (.sum [.predecessor 0 107147 .coefficient, .predecessor 1 107148 .coefficient])

def event107150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26322⟩⟩, .operator (⟨107146, 0⟩, ⟨106992, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩, (1)⟩)

def event107151 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26322⟩⟩, .operator (⟨107146, 2⟩, ⟨106992, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23711⟩⟩]⟩, (-1)⟩)

def event107152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26322⟩⟩) (.sum [.result 107146 .summary, .result 106992 .summary])

def exact107153RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107153RawTermsValid :
    exact107153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107153 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26322⟩⟩) exact107153RawTerms .large 107149 (.finite 1291889174379421642752) (some (107152))

def event107154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26323⟩⟩) 0 ⟨26322⟩ 107153

def event107155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26323⟩⟩) 1 ⟨6680⟩ 5859

def event107156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26323⟩⟩) (.product (.predecessor 0 107154 .coefficient) (.predecessor 1 107155 .coefficient) (⟨false, false, none, none, none⟩))

def event107157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26323⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) [⟨.result 5855 .coefficient, false, none⟩])

def event107158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26323⟩⟩) (.product (.result 107153 .summary) (.transfer 107157) (⟨false, false, none, none, none⟩))

def event107159 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26323⟩⟩, .operator (⟨107153, 0⟩, ⟨5859, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩)

def event107160 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26323⟩⟩, .operator (⟨107153, 1⟩, ⟨5859, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (-1)⟩)

def event107161 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26323⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6679⟩⟩) ⟨6611⟩ 5852)

def event107162 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26323⟩⟩, .relation 107161 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact107163RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107163RawTermsValid :
    exact107163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26323⟩⟩) exact107163RawTerms .large 107156 (.finite 4741253940199267499646124032) (some (107158))

def event107164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6623⟩⟩) 0 ⟨6378⟩ 723

def event107165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6623⟩⟩) 1 ⟨6564⟩ 32

def event107166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6623⟩⟩) (.tensor (.predecessor 0 107164 .coefficient) (.predecessor 1 107165 .coefficient) true false)

def event107167 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6623⟩⟩, .operator (⟨723, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact107168RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact107168RawTermsValid :
    exact107168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6623⟩⟩) exact107168RawTerms .large 107166 .exactZero (none)

def event107169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7097⟩⟩) 0 ⟨5506⟩ 27

def event107170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7097⟩⟩) 1 ⟨6760⟩ 5873

def event107171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7097⟩⟩) (.product (.predecessor 0 107169 .coefficient) (.predecessor 1 107170 .coefficient) (⟨false, false, none, none, none⟩))

def event107172 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7097⟩⟩, .operator (⟨27, 0⟩, ⟨5873, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩)

def exact107173RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩]

theorem exact107173RawTermsValid :
    exact107173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107173 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7097⟩⟩) exact107173RawTerms .large 107171 .exactZero (none)

def event107174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7731⟩⟩) 0 ⟨7097⟩ 107173

def event107175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7731⟩⟩) 1 ⟨6623⟩ 107168

def event107176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7731⟩⟩) (.sum [.predecessor 0 107174 .coefficient, .predecessor 1 107175 .coefficient])

def exact107177RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107177RawTermsValid :
    exact107177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107177 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7731⟩⟩) exact107177RawTerms .large 107176 .exactZero (none)

def event107178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7732⟩⟩) 0 ⟨7731⟩ 107177

def event107179 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7732⟩⟩) 1 ⟨74⟩ 20908

def event107180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7732⟩⟩) (.sum [.predecessor 0 107178 .coefficient, .predecessor 1 107179 .coefficient])

def event107181 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7732⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) [⟨.result 20908 .coefficient, false, none⟩])

def event107182 : Event := .survivorFold (1) 107181

def exact107183RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107183RawTermsValid :
    exact107183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107183 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7732⟩⟩) exact107183RawTerms .large 107180 (.finite 26) (some (107181))

def event107184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7805⟩⟩) 0 ⟨7732⟩ 107183

def event107185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7805⟩⟩) 1 ⟨7732⟩ 107183

def event107186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7805⟩⟩) (.sum [.predecessor 0 107184 .coefficient, .predecessor 1 107185 .coefficient])

def event107187 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7805⟩⟩, .operator (⟨107183, 1⟩, ⟨107183, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event107188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7805⟩⟩, .operator (⟨107183, 0⟩, ⟨107183, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (-1)⟩)

def event107189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7805⟩⟩) (.sum [.result 107183 .summary, .result 107183 .summary])

def exact107190RawTerms : List Term := []

theorem exact107190RawTermsValid :
    exact107190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107190 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7805⟩⟩) exact107190RawTerms .large 107186 (.finite 52) (some (107189))

def event107191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26324⟩⟩) 0 ⟨7805⟩ 107190

def event107192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26324⟩⟩) 1 ⟨26323⟩ 107163

def event107193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26324⟩⟩) (.sum [.predecessor 0 107191 .coefficient, .predecessor 1 107192 .coefficient])

def event107194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26324⟩⟩) (.sum [.result 107190 .summary, .result 107163 .summary])

def exact107195RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107195RawTermsValid :
    exact107195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26324⟩⟩) exact107195RawTerms .large 107193 (.finite 4741253940199267499646124084) (some (107194))

def event107196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26527⟩⟩) 0 ⟨26324⟩ 107195

def event107197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26527⟩⟩) 1 ⟨26526⟩ 106975

def event107198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26527⟩⟩) (.sum [.predecessor 0 107196 .coefficient, .predecessor 1 107197 .coefficient])

def event107199 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26527⟩⟩) (.sum [.result 107195 .summary, .result 106975 .summary])

def exact107200RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107200RawTermsValid :
    exact107200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107200 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26527⟩⟩) exact107200RawTerms .large 107198 (.finite 9482549007414447334737575988) (some (107199))

def event107201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26744⟩⟩) 0 ⟨26527⟩ 107200

def event107202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26744⟩⟩) 1 ⟨26743⟩ 106787

def event107203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26744⟩⟩) (.sum [.predecessor 0 107201 .coefficient, .predecessor 1 107202 .coefficient])

def event107204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26744⟩⟩) (.sum [.result 107200 .summary, .result 106787 .summary])

def exact107205RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107205RawTermsValid :
    exact107205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26744⟩⟩) exact107205RawTerms .large 107203 (.finite 14223885201645539505274355764) (some (107204))

def event107206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26961⟩⟩) 0 ⟨26744⟩ 107205

def event107207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26961⟩⟩) 1 ⟨26960⟩ 106599

def event107208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26961⟩⟩) (.sum [.predecessor 0 107206 .coefficient, .predecessor 1 107207 .coefficient])

def event107209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26961⟩⟩) (.sum [.result 107205 .summary, .result 106599 .summary])

def exact107210RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107210RawTermsValid :
    exact107210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26961⟩⟩) exact107210RawTerms .large 107208 (.finite 18965303649908456346701791284) (some (107209))

def event107211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27178⟩⟩) 0 ⟨26961⟩ 107210

def event107212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27178⟩⟩) 1 ⟨27177⟩ 106411

def event107213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27178⟩⟩) (.sum [.predecessor 0 107211 .coefficient, .predecessor 1 107212 .coefficient])

def event107214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27178⟩⟩) (.sum [.result 107210 .summary, .result 106411 .summary])

def exact107215RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107215RawTermsValid :
    exact107215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107215 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27178⟩⟩) exact107215RawTerms .large 107213 (.finite 23706886606235022529910538292) (some (107214))

def event107216 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27395⟩⟩) 0 ⟨27178⟩ 107215

def event107217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27395⟩⟩) 1 ⟨27394⟩ 106223

def event107218 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27395⟩⟩) (.sum [.predecessor 0 107216 .coefficient, .predecessor 1 107217 .coefficient])

def event107219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27395⟩⟩) (.sum [.result 107215 .summary, .result 106223 .summary])

def exact107220RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107220RawTermsValid :
    exact107220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27395⟩⟩) exact107220RawTerms .large 107218 (.finite 28448551816593413384009941044) (some (107219))

def event107221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27612⟩⟩) 0 ⟨27395⟩ 107220

def event107222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27612⟩⟩) 1 ⟨27611⟩ 106035

def event107223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27612⟩⟩) (.sum [.predecessor 0 107221 .coefficient, .predecessor 1 107222 .coefficient])

def event107224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27612⟩⟩) (.sum [.result 107220 .summary, .result 106035 .summary])

def exact107225RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107225RawTermsValid :
    exact107225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107225 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27612⟩⟩) exact107225RawTerms .large 107223 (.finite 33190381535015453579890655284) (some (107224))

def event107226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27829⟩⟩) 0 ⟨27612⟩ 107225

def event107227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27829⟩⟩) 1 ⟨27828⟩ 105847

def event107228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27829⟩⟩) (.sum [.predecessor 0 107226 .coefficient, .predecessor 1 107227 .coefficient])

def event107229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27829⟩⟩) (.sum [.result 107225 .summary, .result 105847 .summary])

def exact107230RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107230RawTermsValid :
    exact107230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27829⟩⟩) exact107230RawTerms .large 107228 (.finite 37932293507469318446662025268) (some (107229))

def event107231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28046⟩⟩) 0 ⟨27829⟩ 107230

def event107232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28046⟩⟩) 1 ⟨28045⟩ 105659

def event107233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28046⟩⟩) (.sum [.predecessor 0 107231 .coefficient, .predecessor 1 107232 .coefficient])

def event107234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28046⟩⟩) (.sum [.result 107230 .summary, .result 105659 .summary])

def exact107235RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107235RawTermsValid :
    exact107235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28046⟩⟩) exact107235RawTerms .large 107233 (.finite 42674369987986832655214706740) (some (107234))

def event107236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28263⟩⟩) 0 ⟨28046⟩ 107235

def event107237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28263⟩⟩) 1 ⟨28262⟩ 105471

def event107238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28263⟩⟩) (.sum [.predecessor 0 107236 .coefficient, .predecessor 1 107237 .coefficient])

def event107239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28263⟩⟩) (.sum [.result 107235 .summary, .result 105471 .summary])

def exact107240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107240RawTermsValid :
    exact107240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28263⟩⟩) exact107240RawTerms .large 107238 (.finite 47416693230599820876439355444) (some (107239))

def event107241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28480⟩⟩) 0 ⟨28263⟩ 107240

def event107242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28480⟩⟩) 1 ⟨28479⟩ 105283

def event107243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28480⟩⟩) (.sum [.predecessor 0 107241 .coefficient, .predecessor 1 107242 .coefficient])

def event107244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28480⟩⟩) (.sum [.result 107240 .summary, .result 105283 .summary])

def exact107245RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107245RawTermsValid :
    exact107245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107245 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28480⟩⟩) exact107245RawTerms .large 107243 (.finite 52159098727244633768554659892) (some (107244))

def event107246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28697⟩⟩) 0 ⟨28480⟩ 107245

def event107247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28697⟩⟩) 1 ⟨28696⟩ 105095

def event107248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28697⟩⟩) (.sum [.predecessor 0 107246 .coefficient, .predecessor 1 107247 .coefficient])

def event107249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28697⟩⟩) (.sum [.result 107245 .summary, .result 105095 .summary])

def exact107250RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107250RawTermsValid :
    exact107250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28697⟩⟩) exact107250RawTerms .large 107248 (.finite 56901750985984920673341931572) (some (107249))

def event107251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28914⟩⟩) 0 ⟨28697⟩ 107250

def event107252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28914⟩⟩) 1 ⟨28913⟩ 104907

def event107253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28914⟩⟩) (.sum [.predecessor 0 107251 .coefficient, .predecessor 1 107252 .coefficient])

def event107254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28914⟩⟩) (.sum [.result 107250 .summary, .result 104907 .summary])

def exact107255RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107255RawTermsValid :
    exact107255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107255 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28914⟩⟩) exact107255RawTerms .large 107253 (.finite 61644567752788856919910514740) (some (107254))

def event107256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29131⟩⟩) 0 ⟨28914⟩ 107255

def event107257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29131⟩⟩) 1 ⟨29130⟩ 104719

def event107258 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29131⟩⟩) (.sum [.predecessor 0 107256 .coefficient, .predecessor 1 107257 .coefficient])

def event107259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29131⟩⟩) (.sum [.result 107255 .summary, .result 104719 .summary])

def exact107260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17540⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact107260RawTermsValid :
    exact107260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29131⟩⟩) exact107260RawTerms .large 107258 (.finite 66387466773624617837369753652) (some (107259))

def event107261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29348⟩⟩) 0 ⟨29131⟩ 107260

def event107262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29348⟩⟩) 1 ⟨29347⟩ 104531

def event107263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29348⟩⟩) (.sum [.predecessor 0 107261 .coefficient, .predecessor 1 107262 .coefficient])

def eventLeaf6688 : Array AnnotatedEvent := #[
  { event := event107008
    frameStart := 107006 },
  { event := event107009
    frameStart := 107006 },
  { event := event107010
    frameStart := 107006 },
  { event := event107011
    frameStart := 107006 },
  { event := event107012
    frameStart := 107006 },
  { event := event107013
    frameStart := 107006 },
  { event := event107014
    frameStart := 107006 },
  { event := event107015
    frameStart := 107006 },
  { event := event107016
    frameStart := 107006 },
  { event := event107017
    frameStart := 107006 },
  { event := event107018
    frameStart := 107006 },
  { event := event107019
    frameStart := 107006 },
  { event := event107020
    frameStart := 107006 },
  { event := event107021
    frameStart := 107006 },
  { event := event107022
    frameStart := 107006 },
  { event := event107023
    frameStart := 107006 }
]

def eventLeaf6689 : Array AnnotatedEvent := #[
  { event := event107024
    frameStart := 107006 },
  { event := event107025
    frameStart := 107006 },
  { event := event107026
    frameStart := 107006 },
  { event := event107027
    frameStart := 107006 },
  { event := event107028
    frameStart := 107006 },
  { event := event107029
    frameStart := 107006 },
  { event := event107030
    frameStart := 107006 },
  { event := event107031
    frameStart := 107006 },
  { event := event107032
    frameStart := 107006 },
  { event := event107033
    frameStart := 107006 },
  { event := event107034
    frameStart := 107006 },
  { event := event107035
    frameStart := 107006 },
  { event := event107036
    frameStart := 107006 },
  { event := event107037
    frameStart := 107006 },
  { event := event107038
    frameStart := 107006 },
  { event := event107039
    frameStart := 107006 }
]

def eventLeaf6690 : Array AnnotatedEvent := #[
  { event := event107040
    frameStart := 107006 },
  { event := event107041
    frameStart := 107006 },
  { event := event107042
    frameStart := 107006 },
  { event := event107043
    frameStart := 107006 },
  { event := event107044
    frameStart := 107006 },
  { event := event107045
    frameStart := 107006 },
  { event := event107046
    frameStart := 107006 },
  { event := event107047
    frameStart := 107006 },
  { event := event107048
    frameStart := 107048 },
  { event := event107049
    frameStart := 107048 },
  { event := event107050
    frameStart := 107048 },
  { event := event107051
    frameStart := 107048 },
  { event := event107052
    frameStart := 107048 },
  { event := event107053
    frameStart := 107048 },
  { event := event107054
    frameStart := 107048 },
  { event := event107055
    frameStart := 107048 }
]

def eventLeaf6691 : Array AnnotatedEvent := #[
  { event := event107056
    frameStart := 107048 },
  { event := event107057
    frameStart := 107048 },
  { event := event107058
    frameStart := 107048 },
  { event := event107059
    frameStart := 107048 },
  { event := event107060
    frameStart := 107048 },
  { event := event107061
    frameStart := 107048 },
  { event := event107062
    frameStart := 107048 },
  { event := event107063
    frameStart := 107048 },
  { event := event107064
    frameStart := 107048 },
  { event := event107065
    frameStart := 107048 },
  { event := event107066
    frameStart := 107048 },
  { event := event107067
    frameStart := 107048 },
  { event := event107068
    frameStart := 107048 },
  { event := event107069
    frameStart := 107048 },
  { event := event107070
    frameStart := 107048 },
  { event := event107071
    frameStart := 107048 }
]

def eventLeaf6692 : Array AnnotatedEvent := #[
  { event := event107072
    frameStart := 107048 },
  { event := event107073
    frameStart := 107048 },
  { event := event107074
    frameStart := 107048 },
  { event := event107075
    frameStart := 107048 },
  { event := event107076
    frameStart := 107048 },
  { event := event107077
    frameStart := 107048 },
  { event := event107078
    frameStart := 107048 },
  { event := event107079
    frameStart := 107048 },
  { event := event107080
    frameStart := 107048 },
  { event := event107081
    frameStart := 107048 },
  { event := event107082
    frameStart := 107048 },
  { event := event107083
    frameStart := 107048 },
  { event := event107084
    frameStart := 107048 },
  { event := event107085
    frameStart := 107048 },
  { event := event107086
    frameStart := 107048 },
  { event := event107087
    frameStart := 107048 }
]

def eventLeaf6693 : Array AnnotatedEvent := #[
  { event := event107088
    frameStart := 107048 },
  { event := event107089
    frameStart := 107048 },
  { event := event107090
    frameStart := 107048 },
  { event := event107091
    frameStart := 107048 },
  { event := event107092
    frameStart := 107048 },
  { event := event107093
    frameStart := 107048 },
  { event := event107094
    frameStart := 107048 },
  { event := event107095
    frameStart := 107048 },
  { event := event107096
    frameStart := 107048 },
  { event := event107097
    frameStart := 107048 },
  { event := event107098
    frameStart := 107048 },
  { event := event107099
    frameStart := 107048 },
  { event := event107100
    frameStart := 107048 },
  { event := event107101
    frameStart := 107048 },
  { event := event107102
    frameStart := 107048 },
  { event := event107103
    frameStart := 107048 }
]

def eventLeaf6694 : Array AnnotatedEvent := #[
  { event := event107104
    frameStart := 107048 },
  { event := event107105
    frameStart := 107048 },
  { event := event107106
    frameStart := 107048 },
  { event := event107107
    frameStart := 107048 },
  { event := event107108
    frameStart := 107048 },
  { event := event107109
    frameStart := 107048 },
  { event := event107110
    frameStart := 107048 },
  { event := event107111
    frameStart := 107048 },
  { event := event107112
    frameStart := 107048 },
  { event := event107113
    frameStart := 107048 },
  { event := event107114
    frameStart := 107048 },
  { event := event107115
    frameStart := 107048 },
  { event := event107116
    frameStart := 107048 },
  { event := event107117
    frameStart := 107048 },
  { event := event107118
    frameStart := 107048 },
  { event := event107119
    frameStart := 107048 }
]

def eventLeaf6695 : Array AnnotatedEvent := #[
  { event := event107120
    frameStart := 107048 },
  { event := event107121
    frameStart := 107048 },
  { event := event107122
    frameStart := 107048 },
  { event := event107123
    frameStart := 107048 },
  { event := event107124
    frameStart := 107048 },
  { event := event107125
    frameStart := 107048 },
  { event := event107126
    frameStart := 107048 },
  { event := event107127
    frameStart := 107048 },
  { event := event107128
    frameStart := 107048 },
  { event := event107129
    frameStart := 107048 },
  { event := event107130
    frameStart := 107048 },
  { event := event107131
    frameStart := 107048 },
  { event := event107132
    frameStart := 107048 },
  { event := event107133
    frameStart := 107048 },
  { event := event107134
    frameStart := 107048 },
  { event := event107135
    frameStart := 107048 }
]

def eventLeaf6696 : Array AnnotatedEvent := #[
  { event := event107136
    frameStart := 107048 },
  { event := event107137
    frameStart := 107048 },
  { event := event107138
    frameStart := 107048 },
  { event := event107139
    frameStart := 107048 },
  { event := event107140
    frameStart := 0 },
  { event := event107141
    frameStart := 0 },
  { event := event107142
    frameStart := 0 },
  { event := event107143
    frameStart := 0 },
  { event := event107144
    frameStart := 0 },
  { event := event107145
    frameStart := 0 },
  { event := event107146
    frameStart := 0 },
  { event := event107147
    frameStart := 0 },
  { event := event107148
    frameStart := 0 },
  { event := event107149
    frameStart := 0 },
  { event := event107150
    frameStart := 0 },
  { event := event107151
    frameStart := 0 }
]

def eventLeaf6697 : Array AnnotatedEvent := #[
  { event := event107152
    frameStart := 0 },
  { event := event107153
    frameStart := 0 },
  { event := event107154
    frameStart := 0 },
  { event := event107155
    frameStart := 0 },
  { event := event107156
    frameStart := 0 },
  { event := event107157
    frameStart := 0 },
  { event := event107158
    frameStart := 0 },
  { event := event107159
    frameStart := 0 },
  { event := event107160
    frameStart := 0 },
  { event := event107161
    frameStart := 0 },
  { event := event107162
    frameStart := 0 },
  { event := event107163
    frameStart := 0 },
  { event := event107164
    frameStart := 0 },
  { event := event107165
    frameStart := 0 },
  { event := event107166
    frameStart := 0 },
  { event := event107167
    frameStart := 0 }
]

def eventLeaf6698 : Array AnnotatedEvent := #[
  { event := event107168
    frameStart := 0 },
  { event := event107169
    frameStart := 0 },
  { event := event107170
    frameStart := 0 },
  { event := event107171
    frameStart := 0 },
  { event := event107172
    frameStart := 0 },
  { event := event107173
    frameStart := 0 },
  { event := event107174
    frameStart := 0 },
  { event := event107175
    frameStart := 0 },
  { event := event107176
    frameStart := 0 },
  { event := event107177
    frameStart := 0 },
  { event := event107178
    frameStart := 0 },
  { event := event107179
    frameStart := 0 },
  { event := event107180
    frameStart := 0 },
  { event := event107181
    frameStart := 0 },
  { event := event107182
    frameStart := 0 },
  { event := event107183
    frameStart := 0 }
]

def eventLeaf6699 : Array AnnotatedEvent := #[
  { event := event107184
    frameStart := 0 },
  { event := event107185
    frameStart := 0 },
  { event := event107186
    frameStart := 0 },
  { event := event107187
    frameStart := 0 },
  { event := event107188
    frameStart := 0 },
  { event := event107189
    frameStart := 0 },
  { event := event107190
    frameStart := 0 },
  { event := event107191
    frameStart := 0 },
  { event := event107192
    frameStart := 0 },
  { event := event107193
    frameStart := 0 },
  { event := event107194
    frameStart := 0 },
  { event := event107195
    frameStart := 0 },
  { event := event107196
    frameStart := 0 },
  { event := event107197
    frameStart := 0 },
  { event := event107198
    frameStart := 0 },
  { event := event107199
    frameStart := 0 }
]

def eventLeaf6700 : Array AnnotatedEvent := #[
  { event := event107200
    frameStart := 0 },
  { event := event107201
    frameStart := 0 },
  { event := event107202
    frameStart := 0 },
  { event := event107203
    frameStart := 0 },
  { event := event107204
    frameStart := 0 },
  { event := event107205
    frameStart := 0 },
  { event := event107206
    frameStart := 0 },
  { event := event107207
    frameStart := 0 },
  { event := event107208
    frameStart := 0 },
  { event := event107209
    frameStart := 0 },
  { event := event107210
    frameStart := 0 },
  { event := event107211
    frameStart := 0 },
  { event := event107212
    frameStart := 0 },
  { event := event107213
    frameStart := 0 },
  { event := event107214
    frameStart := 0 },
  { event := event107215
    frameStart := 0 }
]

def eventLeaf6701 : Array AnnotatedEvent := #[
  { event := event107216
    frameStart := 0 },
  { event := event107217
    frameStart := 0 },
  { event := event107218
    frameStart := 0 },
  { event := event107219
    frameStart := 0 },
  { event := event107220
    frameStart := 0 },
  { event := event107221
    frameStart := 0 },
  { event := event107222
    frameStart := 0 },
  { event := event107223
    frameStart := 0 },
  { event := event107224
    frameStart := 0 },
  { event := event107225
    frameStart := 0 },
  { event := event107226
    frameStart := 0 },
  { event := event107227
    frameStart := 0 },
  { event := event107228
    frameStart := 0 },
  { event := event107229
    frameStart := 0 },
  { event := event107230
    frameStart := 0 },
  { event := event107231
    frameStart := 0 }
]

def eventLeaf6702 : Array AnnotatedEvent := #[
  { event := event107232
    frameStart := 0 },
  { event := event107233
    frameStart := 0 },
  { event := event107234
    frameStart := 0 },
  { event := event107235
    frameStart := 0 },
  { event := event107236
    frameStart := 0 },
  { event := event107237
    frameStart := 0 },
  { event := event107238
    frameStart := 0 },
  { event := event107239
    frameStart := 0 },
  { event := event107240
    frameStart := 0 },
  { event := event107241
    frameStart := 0 },
  { event := event107242
    frameStart := 0 },
  { event := event107243
    frameStart := 0 },
  { event := event107244
    frameStart := 0 },
  { event := event107245
    frameStart := 0 },
  { event := event107246
    frameStart := 0 },
  { event := event107247
    frameStart := 0 }
]

def eventLeaf6703 : Array AnnotatedEvent := #[
  { event := event107248
    frameStart := 0 },
  { event := event107249
    frameStart := 0 },
  { event := event107250
    frameStart := 0 },
  { event := event107251
    frameStart := 0 },
  { event := event107252
    frameStart := 0 },
  { event := event107253
    frameStart := 0 },
  { event := event107254
    frameStart := 0 },
  { event := event107255
    frameStart := 0 },
  { event := event107256
    frameStart := 0 },
  { event := event107257
    frameStart := 0 },
  { event := event107258
    frameStart := 0 },
  { event := event107259
    frameStart := 0 },
  { event := event107260
    frameStart := 0 },
  { event := event107261
    frameStart := 0 },
  { event := event107262
    frameStart := 0 },
  { event := event107263
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events418
