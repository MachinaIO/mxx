import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events418

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event107008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event107009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41470⟩⟩) 0 ⟨40117⟩ 106995

def event107010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41470⟩⟩) 1 ⟨136⟩ 107008

def event107011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41470⟩⟩) (.sum [.predecessor 0 107009 .coefficient, .predecessor 1 107010 .coefficient])

def event107012 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41470⟩⟩) (.finite 46)

def event107013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41471⟩⟩) 0 ⟨41470⟩ 107012

def event107014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41471⟩⟩) (.identity (.predecessor 0 107013 .coefficient))

def exact107015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], []⟩, (1)⟩]

theorem exact107015RawTermsValid :
    exact107015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41471⟩⟩) exact107015RawTerms (.finite 46) 107014 .exactZero (none)

def event107016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact107017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact107017RawTermsValid :
    exact107017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact107017RawTerms .large 107016 .exactZero (none)

def event107018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41472⟩⟩) 0 ⟨6908⟩ 107017

def event107019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41472⟩⟩) 1 ⟨41471⟩ 107015

def event107020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41472⟩⟩) (.product (.predecessor 0 107018 .coefficient) (.predecessor 1 107019 .coefficient) (⟨false, false, none, none, none⟩))

def event107021 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41472⟩⟩, .operator (⟨107017, 0⟩, ⟨107015, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact107022RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact107022RawTermsValid :
    exact107022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41472⟩⟩) exact107022RawTerms .large 107020 .exactZero (none)

def event107023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 106999

def event107024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact107025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact107025RawTermsValid :
    exact107025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact107025RawTerms .large 107024 .exactZero (none)

def event107026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41473⟩⟩) 0 ⟨7193⟩ 107025

def event107027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41473⟩⟩) 1 ⟨41472⟩ 107022

def event107028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41473⟩⟩) (.sum [.predecessor 0 107026 .coefficient, .predecessor 1 107027 .coefficient])

def exact107029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107029RawTermsValid :
    exact107029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41473⟩⟩) exact107029RawTerms .large 107028 .exactZero (none)

def event107030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42015⟩⟩) 0 ⟨41473⟩ 107029

def event107031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42015⟩⟩) 1 ⟨42014⟩ 107006

def event107032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42015⟩⟩) (.product (.predecessor 0 107030 .coefficient) (.predecessor 1 107031 .coefficient) (⟨false, false, none, none, none⟩))

def event107033 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42015⟩⟩, .operator (⟨107029, 0⟩, ⟨107006, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩)

def event107034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42015⟩⟩, .operator (⟨107029, 1⟩, ⟨107006, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩)

def event107035 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42015⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42014⟩⟩) ⟨41270⟩ 107003)

def event107036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42015⟩⟩, .relation 107035 0, ⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (-1)⟩)

def exact107037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (-1)⟩]

theorem exact107037RawTermsValid :
    exact107037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42015⟩⟩) exact107037RawTerms .large 107032 .exactZero (none)

def event107038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40332⟩⟩) 0 ⟨40117⟩ 106995

def event107039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40332⟩⟩) (.authority (.programFamilyFact))

def exact107040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], []⟩, (1)⟩]

theorem exact107040RawTermsValid :
    exact107040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40332⟩⟩) exact107040RawTerms (.finite 63) 107039 .exactZero (none)

def event107041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40333⟩⟩) 0 ⟨6908⟩ 107017

def event107042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40333⟩⟩) 1 ⟨40332⟩ 107040

def event107043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40333⟩⟩) (.product (.predecessor 0 107041 .coefficient) (.predecessor 1 107042 .coefficient) (⟨false, true, none, none, some 1⟩))

def event107044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40333⟩⟩, .operator (⟨107017, 0⟩, ⟨107040, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact107045RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact107045RawTermsValid :
    exact107045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107045 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40333⟩⟩) exact107045RawTerms .large 107043 .exactZero (none)

def event107046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 106999

def event107047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact107048RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact107048RawTermsValid :
    exact107048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact107048RawTerms .large 107047 .exactZero (none)

def event107049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40334⟩⟩) 0 ⟨7226⟩ 107048

def event107050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40334⟩⟩) 1 ⟨40333⟩ 107045

def event107051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40334⟩⟩) (.sum [.predecessor 0 107049 .coefficient, .predecessor 1 107050 .coefficient])

def exact107052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107052RawTermsValid :
    exact107052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40334⟩⟩) exact107052RawTerms .large 107051 .exactZero (none)

def event107053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42018⟩⟩) 0 ⟨40334⟩ 107052

def event107054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42018⟩⟩) 1 ⟨42015⟩ 107037

def event107055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42018⟩⟩) (.sum [.predecessor 0 107053 .coefficient, .predecessor 1 107054 .coefficient])

def exact107056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107056RawTermsValid :
    exact107056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42018⟩⟩) exact107056RawTerms .large 107055 .exactZero (none)

def event107057 : Event := .preFoldPolynomial 107056 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact107058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event107058 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨42018⟩⟩) 107057 exact107058RawTerms .large 107055 .exactZero (none)

def event107059 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40117⟩⟩) ⟨⟨105⟩, ⟨87⟩, ⟨135⟩⟩ ⟨106901, 107059⟩

def event107060 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40879⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩) (1) 0 2 (.universal 107059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40876⟩⟩]⟩) (none) 107058)

def event107061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40879⟩⟩, .relation 107060 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩)

def event107062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40879⟩⟩, .relation 107060 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩)

def event107063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40879⟩⟩, .relation 107060 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩)

def event107064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40879⟩⟩, .relation 107060 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact107065RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107065RawTermsValid :
    exact107065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40879⟩⟩) exact107065RawTerms .large 106897 (.finite 202072841853861888) (some (106899))

def event107066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42017⟩⟩) 0 ⟨40879⟩ 107065

def event107067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42017⟩⟩) 1 ⟨42016⟩ 106887

def event107068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42017⟩⟩) (.sum [.predecessor 0 107066 .coefficient, .predecessor 1 107067 .coefficient])

def event107069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42017⟩⟩, .operator (⟨107065, 0⟩, ⟨106887, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩)

def event107070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42017⟩⟩, .operator (⟨107065, 2⟩, ⟨106887, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40116⟩⟩], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (-1)⟩)

def event107071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42017⟩⟩) (.sum [.result 107065 .summary, .result 106887 .summary])

def exact107072RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨40332⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107072RawTermsValid :
    exact107072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42017⟩⟩) exact107072RawTerms .large 107068 (.finite 32193129122288829188810200055808) (some (107071))

def event107073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38588⟩⟩) 0 ⟨37437⟩ 4691

def event107074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38588⟩⟩) (.authority (.programFamilyFact))

def event107075 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38588⟩⟩) (.finite 3720)

def event107076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38590⟩⟩) 0 ⟨7177⟩ 15500

def event107077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38590⟩⟩) 1 ⟨38588⟩ 107075

def event107078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38590⟩⟩) (.authority (.operator))

def exact107079RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38590⟩⟩]⟩, (1)⟩]

theorem exact107079RawTermsValid :
    exact107079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107079 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38590⟩⟩) exact107079RawTerms .large 107078 .exactZero (none)

def event107080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39334⟩⟩) 0 ⟨38590⟩ 107079

def event107081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39334⟩⟩) (.authority (.operator))

def exact107082RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39334⟩⟩]⟩, (1)⟩]

theorem exact107082RawTermsValid :
    exact107082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39334⟩⟩) exact107082RawTerms (.finite 8192) 107081 .exactZero (none)

def event107083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38434⟩⟩) 0 ⟨37140⟩ 4685

def event107084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38434⟩⟩) (.authority (.programFamilyFact))

def event107085 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38434⟩⟩) (.finite 3720)

def event107086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38435⟩⟩) 0 ⟨7177⟩ 15500

def event107087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38435⟩⟩) 1 ⟨38434⟩ 107085

def event107088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38435⟩⟩) (.authority (.operator))

def exact107089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩, (1)⟩]

theorem exact107089RawTermsValid :
    exact107089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107089 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38435⟩⟩) exact107089RawTerms .large 107088 .exactZero (none)

def event107090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38950⟩⟩) 0 ⟨38435⟩ 107089

def event107091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38950⟩⟩) (.authority (.operator))

def exact107092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩, (1)⟩]

theorem exact107092RawTermsValid :
    exact107092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38950⟩⟩) exact107092RawTerms (.finite 8192) 107091 .exactZero (none)

def event107093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37141⟩⟩) 0 ⟨37138⟩ 4674

def event107094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37141⟩⟩) 1 ⟨6992⟩ 105153

def event107095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37141⟩⟩) (.tensor (.predecessor 0 107093 .coefficient) (.predecessor 1 107094 .coefficient) true false)

def event107096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37141⟩⟩, .operator (⟨4674, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact107097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact107097RawTermsValid :
    exact107097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37141⟩⟩) exact107097RawTerms .large 107095 .exactZero (none)

def event107098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8701⟩⟩) 0 ⟨5768⟩ 105023

def event107099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8701⟩⟩) 1 ⟨7281⟩ 19084

def event107100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8701⟩⟩) (.product (.predecessor 0 107098 .coefficient) (.predecessor 1 107099 .coefficient) (⟨false, false, none, none, none⟩))

def event107101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8701⟩⟩, .operator (⟨105023, 0⟩, ⟨19084, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact107102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact107102RawTermsValid :
    exact107102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8701⟩⟩) exact107102RawTerms .large 107100 .exactZero (none)

def event107103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37142⟩⟩) 0 ⟨8701⟩ 107102

def event107104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37142⟩⟩) 1 ⟨37141⟩ 107097

def event107105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37142⟩⟩) (.sum [.predecessor 0 107103 .coefficient, .predecessor 1 107104 .coefficient])

def exact107106RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107106RawTermsValid :
    exact107106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37142⟩⟩) exact107106RawTerms .large 107105 .exactZero (none)

def event107107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37143⟩⟩) 0 ⟨37142⟩ 107106

def event107108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37143⟩⟩) 1 ⟨107⟩ 19076

def event107109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37143⟩⟩) (.sum [.predecessor 0 107107 .coefficient, .predecessor 1 107108 .coefficient])

def event107110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37143⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩) [⟨.result 19076 .coefficient, false, none⟩])

def event107111 : Event := .survivorFold (1) 107110

def exact107112RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107112RawTermsValid :
    exact107112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37143⟩⟩) exact107112RawTerms .large 107109 (.finite 26) (some (107110))

def event107113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37144⟩⟩) 0 ⟨37143⟩ 107112

def event107114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37144⟩⟩) 1 ⟨13896⟩ 4677

def event107115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37144⟩⟩) (.product (.predecessor 0 107113 .coefficient) (.predecessor 1 107114 .coefficient) (⟨false, true, none, none, some 1⟩))

def event107116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37144⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩], []⟩) [⟨.result 4677 .coefficient, true, some 1⟩])

def event107117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37144⟩⟩) (.product (.result 107112 .summary) (.transfer 107116) (⟨false, false, none, none, none⟩))

def event107118 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37144⟩⟩, .operator (⟨107112, 1⟩, ⟨4677, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event107119 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37144⟩⟩, .operator (⟨107112, 0⟩, ⟨4677, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact107120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107120RawTermsValid :
    exact107120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37144⟩⟩) exact107120RawTerms .large 107115 (.finite 35782656) (some (107117))

def event107121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13897⟩⟩) 0 ⟨13896⟩ 4677

def event107122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13897⟩⟩) 1 ⟨6992⟩ 105153

def event107123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13897⟩⟩) (.tensor (.predecessor 0 107121 .coefficient) (.predecessor 1 107122 .coefficient) true false)

def event107124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13897⟩⟩, .operator (⟨4677, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact107125RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact107125RawTermsValid :
    exact107125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13897⟩⟩) exact107125RawTerms .large 107123 .exactZero (none)

def event107126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8718⟩⟩) 0 ⟨5768⟩ 105023

def event107127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8718⟩⟩) 1 ⟨7298⟩ 19125

def event107128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8718⟩⟩) (.product (.predecessor 0 107126 .coefficient) (.predecessor 1 107127 .coefficient) (⟨false, false, none, none, none⟩))

def event107129 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8718⟩⟩, .operator (⟨105023, 0⟩, ⟨19125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩)

def exact107130RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact107130RawTermsValid :
    exact107130RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107130 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8718⟩⟩) exact107130RawTerms .large 107128 .exactZero (none)

def event107131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13898⟩⟩) 0 ⟨8718⟩ 107130

def event107132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13898⟩⟩) 1 ⟨13897⟩ 107125

def event107133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13898⟩⟩) (.sum [.predecessor 0 107131 .coefficient, .predecessor 1 107132 .coefficient])

def exact107134RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107134RawTermsValid :
    exact107134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13898⟩⟩) exact107134RawTerms .large 107133 .exactZero (none)

def event107135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13899⟩⟩) 0 ⟨13898⟩ 107134

def event107136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13899⟩⟩) 1 ⟨124⟩ 19117

def event107137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13899⟩⟩) (.sum [.predecessor 0 107135 .coefficient, .predecessor 1 107136 .coefficient])

def event107138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13899⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩) [⟨.result 19117 .coefficient, false, none⟩])

def event107139 : Event := .survivorFold (1) 107138

def exact107140RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107140RawTermsValid :
    exact107140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13899⟩⟩) exact107140RawTerms .large 107137 (.finite 26) (some (107138))

def event107141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13900⟩⟩) 0 ⟨13899⟩ 107140

def event107142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13900⟩⟩) 1 ⟨9554⟩ 19114

def event107143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13900⟩⟩) (.product (.predecessor 0 107141 .coefficient) (.predecessor 1 107142 .coefficient) (⟨false, false, none, none, none⟩))

def event107144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13900⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) [⟨.result 19110 .coefficient, false, none⟩])

def event107145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13900⟩⟩) (.product (.result 107140 .summary) (.transfer 107144) (⟨false, false, none, none, none⟩))

def event107146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13900⟩⟩, .operator (⟨107140, 1⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (-1)⟩)

def event107147 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13900⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084)

def event107148 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13900⟩⟩, .relation 107147 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩)

def event107149 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13900⟩⟩, .operator (⟨107140, 0⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact107150RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩]

theorem exact107150RawTermsValid :
    exact107150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13900⟩⟩) exact107150RawTerms .large 107143 (.finite 279172874240) (some (107145))

def event107151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37145⟩⟩) 0 ⟨13900⟩ 107150

def event107152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37145⟩⟩) 1 ⟨37144⟩ 107120

def event107153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37145⟩⟩) (.sum [.predecessor 0 107151 .coefficient, .predecessor 1 107152 .coefficient])

def event107154 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37145⟩⟩, .operator (⟨107150, 1⟩, ⟨107120, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def event107155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37145⟩⟩) (.sum [.result 107150 .summary, .result 107120 .summary])

def exact107156RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact107156RawTermsValid :
    exact107156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37145⟩⟩) exact107156RawTerms .large 107153 (.finite 279208656896) (some (107155))

def event107157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38951⟩⟩) 0 ⟨37145⟩ 107156

def event107158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38951⟩⟩) 1 ⟨38950⟩ 107092

def event107159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38951⟩⟩) (.product (.predecessor 0 107157 .coefficient) (.predecessor 1 107158 .coefficient) (⟨false, false, none, none, none⟩))

def event107160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38951⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) [⟨.result 107092 .coefficient, false, none⟩])

def event107161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38951⟩⟩) (.product (.result 107156 .summary) (.transfer 107160) (⟨false, false, none, none, none⟩))

def event107162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38951⟩⟩, .operator (⟨107156, 1⟩, ⟨107092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩, (-1)⟩)

def event107163 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38951⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38950⟩⟩) ⟨38435⟩ 107089)

def event107164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38951⟩⟩, .relation 107163 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩, (-1)⟩)

def event107165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38951⟩⟩, .operator (⟨107156, 0⟩, ⟨107092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩, (1)⟩)

def exact107166RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], [⟨.program ⟨257⟩, ⟨38435⟩⟩]⟩, (-1)⟩]

theorem exact107166RawTermsValid :
    exact107166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38951⟩⟩) exact107166RawTerms .large 107159 (.finite 2997980125321012183040) (some (107161))

def event107167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37879⟩⟩) 0 ⟨37140⟩ 4685

def event107168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37879⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact107169RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩]

theorem exact107169RawTermsValid :
    exact107169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37879⟩⟩) exact107169RawTerms (.finite 5647228698) 107168 .exactZero (none)

def event107170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37881⟩⟩) 0 ⟨37879⟩ 107169

def event107171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37881⟩⟩) 1 ⟨2370⟩ 4

def event107172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37881⟩⟩) (.scale (.predecessor 0 107170 .coefficient) (.value (.predecessor 1 107171 .coefficient)))

def exact107173RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩]

theorem exact107173RawTermsValid :
    exact107173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37881⟩⟩) exact107173RawTerms (.finite 5647228698) 107172 .exactZero (none)

def event107174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37882⟩⟩) 0 ⟨5770⟩ 105245

def event107175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37882⟩⟩) 1 ⟨37881⟩ 107173

def event107176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37882⟩⟩) (.product (.predecessor 0 107174 .coefficient) (.predecessor 1 107175 .coefficient) (⟨false, false, none, none, none⟩))

def event107177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37882⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩) [⟨.result 107169 .coefficient, false, none⟩])

def event107178 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37882⟩⟩) (.product (.result 105245 .summary) (.transfer 107177) (⟨false, false, none, none, none⟩))

def event107179 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37882⟩⟩, .operator (⟨105245, 0⟩, ⟨107173, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩)

def event107180 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨37880⟩⟩)

def event107181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event107182 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event107183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event107184 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event107185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event107186 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event107187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event107188 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event107189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 107188

def event107190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 107186

def event107191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 107189 .coefficient) (.value (.predecessor 1 107190 .coefficient)))

def event107192 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event107193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 107192

def event107194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 107184

def event107195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 107193 .coefficient, .predecessor 1 107194 .coefficient])

def event107196 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event107197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 107196

def event107198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 107182

def event107199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 107198 .coefficient))

def event107200 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event107201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37138⟩⟩) 0 ⟨5766⟩ 107200

def event107202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37138⟩⟩) (.authority (.programFamilyFact))

def exact107203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩, (1)⟩]

theorem exact107203RawTermsValid :
    exact107203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37138⟩⟩) exact107203RawTerms (.finite 42) 107202 .exactZero (none)

def event107204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13896⟩⟩) 0 ⟨5766⟩ 107200

def event107205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13896⟩⟩) (.authority (.programFamilyFact))

def exact107206RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩], []⟩, (1)⟩]

theorem exact107206RawTermsValid :
    exact107206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107206 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13896⟩⟩) exact107206RawTerms (.finite 42) 107205 .exactZero (none)

def event107207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37139⟩⟩) 0 ⟨13896⟩ 107206

def event107208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37139⟩⟩) 1 ⟨37138⟩ 107203

def event107209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37139⟩⟩) (.product (.predecessor 0 107207 .coefficient) (.predecessor 1 107208 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event107210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37139⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩) [⟨.result 107206 .coefficient, true, some 1⟩, ⟨.result 107203 .coefficient, true, some 1⟩])

def event107211 : Event := .survivorFold (1) 107210

def exact107212RawTerms : List Term := []

theorem exact107212RawTermsValid :
    exact107212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37139⟩⟩) exact107212RawTerms (.finite 1764) 107209 (.finite 1764) (some (107210))

def event107213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37140⟩⟩) 0 ⟨37139⟩ 107212

def event107214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37140⟩⟩) (.identity (.predecessor 0 107213 .coefficient))

def event107215 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37140⟩⟩) (.finite 1764)

def event107216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37879⟩⟩) 0 ⟨37140⟩ 107215

def event107217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37879⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact107218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩]

theorem exact107218RawTermsValid :
    exact107218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37879⟩⟩) exact107218RawTerms (.finite 5647228698) 107217 .exactZero (none)

def event107219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact107220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact107220RawTermsValid :
    exact107220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact107220RawTerms .large 107219 .exactZero (none)

def event107221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37880⟩⟩) 0 ⟨35⟩ 107220

def event107222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37880⟩⟩) 1 ⟨37879⟩ 107218

def event107223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37880⟩⟩) (.product (.predecessor 0 107221 .coefficient) (.predecessor 1 107222 .coefficient) (⟨false, false, none, none, none⟩))

def event107224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37880⟩⟩, .operator (⟨107220, 0⟩, ⟨107218, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩)

def exact107225RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩]

theorem exact107225RawTermsValid :
    exact107225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37880⟩⟩) exact107225RawTerms .large 107223 .exactZero (none)

def event107226 : Event := .preFoldPolynomial 107225 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩] .exactZero none

def exact107227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37879⟩⟩]⟩, (1)⟩]

def event107227 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨37880⟩⟩) 107226 exact107227RawTerms .large 107223 .exactZero (none)

def event107228 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38954⟩⟩)

def event107229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event107230 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event107231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event107232 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event107233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event107234 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event107235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event107236 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event107237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 107236

def event107238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 107234

def event107239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 107237 .coefficient) (.value (.predecessor 1 107238 .coefficient)))

def event107240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event107241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 107240

def event107242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 107232

def event107243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 107241 .coefficient, .predecessor 1 107242 .coefficient])

def event107244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event107245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 107244

def event107246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 107230

def event107247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 107246 .coefficient))

def event107248 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event107249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37138⟩⟩) 0 ⟨5766⟩ 107248

def event107250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37138⟩⟩) (.authority (.programFamilyFact))

def exact107251RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩, (1)⟩]

theorem exact107251RawTermsValid :
    exact107251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37138⟩⟩) exact107251RawTerms (.finite 42) 107250 .exactZero (none)

def event107252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13896⟩⟩) 0 ⟨5766⟩ 107248

def event107253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13896⟩⟩) (.authority (.programFamilyFact))

def exact107254RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩], []⟩, (1)⟩]

theorem exact107254RawTermsValid :
    exact107254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13896⟩⟩) exact107254RawTerms (.finite 42) 107253 .exactZero (none)

def event107255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37139⟩⟩) 0 ⟨13896⟩ 107254

def event107256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37139⟩⟩) 1 ⟨37138⟩ 107251

def event107257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37139⟩⟩) (.product (.predecessor 0 107255 .coefficient) (.predecessor 1 107256 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event107258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37139⟩⟩, .operator (⟨107254, 0⟩, ⟨107251, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩, (1)⟩)

def exact107259RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13896⟩⟩, ⟨.program ⟨257⟩, ⟨37138⟩⟩], []⟩, (1)⟩]

theorem exact107259RawTermsValid :
    exact107259RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event107259 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37139⟩⟩) exact107259RawTerms (.finite 1764) 107257 .exactZero (none)

def event107260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37140⟩⟩) 0 ⟨37139⟩ 107259

def event107261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37140⟩⟩) (.identity (.predecessor 0 107260 .coefficient))

def event107262 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37140⟩⟩) (.finite 1764)

def event107263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38434⟩⟩) 0 ⟨37140⟩ 107262

def eventLeaf6688 : Array AnnotatedEvent := #[
  { event := event107008
    frameStart := 106955 },
  { event := event107009
    frameStart := 106955 },
  { event := event107010
    frameStart := 106955 },
  { event := event107011
    frameStart := 106955 },
  { event := event107012
    frameStart := 106955 },
  { event := event107013
    frameStart := 106955 },
  { event := event107014
    frameStart := 106955 },
  { event := event107015
    frameStart := 106955 },
  { event := event107016
    frameStart := 106955 },
  { event := event107017
    frameStart := 106955 },
  { event := event107018
    frameStart := 106955 },
  { event := event107019
    frameStart := 106955 },
  { event := event107020
    frameStart := 106955 },
  { event := event107021
    frameStart := 106955 },
  { event := event107022
    frameStart := 106955 },
  { event := event107023
    frameStart := 106955 }
]

def eventLeaf6689 : Array AnnotatedEvent := #[
  { event := event107024
    frameStart := 106955 },
  { event := event107025
    frameStart := 106955 },
  { event := event107026
    frameStart := 106955 },
  { event := event107027
    frameStart := 106955 },
  { event := event107028
    frameStart := 106955 },
  { event := event107029
    frameStart := 106955 },
  { event := event107030
    frameStart := 106955 },
  { event := event107031
    frameStart := 106955 },
  { event := event107032
    frameStart := 106955 },
  { event := event107033
    frameStart := 106955 },
  { event := event107034
    frameStart := 106955 },
  { event := event107035
    frameStart := 106955 },
  { event := event107036
    frameStart := 106955 },
  { event := event107037
    frameStart := 106955 },
  { event := event107038
    frameStart := 106955 },
  { event := event107039
    frameStart := 106955 }
]

def eventLeaf6690 : Array AnnotatedEvent := #[
  { event := event107040
    frameStart := 106955 },
  { event := event107041
    frameStart := 106955 },
  { event := event107042
    frameStart := 106955 },
  { event := event107043
    frameStart := 106955 },
  { event := event107044
    frameStart := 106955 },
  { event := event107045
    frameStart := 106955 },
  { event := event107046
    frameStart := 106955 },
  { event := event107047
    frameStart := 106955 },
  { event := event107048
    frameStart := 106955 },
  { event := event107049
    frameStart := 106955 },
  { event := event107050
    frameStart := 106955 },
  { event := event107051
    frameStart := 106955 },
  { event := event107052
    frameStart := 106955 },
  { event := event107053
    frameStart := 106955 },
  { event := event107054
    frameStart := 106955 },
  { event := event107055
    frameStart := 106955 }
]

def eventLeaf6691 : Array AnnotatedEvent := #[
  { event := event107056
    frameStart := 106955 },
  { event := event107057
    frameStart := 106955 },
  { event := event107058
    frameStart := 106955 },
  { event := event107059
    frameStart := 0 },
  { event := event107060
    frameStart := 0 },
  { event := event107061
    frameStart := 0 },
  { event := event107062
    frameStart := 0 },
  { event := event107063
    frameStart := 0 },
  { event := event107064
    frameStart := 0 },
  { event := event107065
    frameStart := 0 },
  { event := event107066
    frameStart := 0 },
  { event := event107067
    frameStart := 0 },
  { event := event107068
    frameStart := 0 },
  { event := event107069
    frameStart := 0 },
  { event := event107070
    frameStart := 0 },
  { event := event107071
    frameStart := 0 }
]

def eventLeaf6692 : Array AnnotatedEvent := #[
  { event := event107072
    frameStart := 0 },
  { event := event107073
    frameStart := 0 },
  { event := event107074
    frameStart := 0 },
  { event := event107075
    frameStart := 0 },
  { event := event107076
    frameStart := 0 },
  { event := event107077
    frameStart := 0 },
  { event := event107078
    frameStart := 0 },
  { event := event107079
    frameStart := 0 },
  { event := event107080
    frameStart := 0 },
  { event := event107081
    frameStart := 0 },
  { event := event107082
    frameStart := 0 },
  { event := event107083
    frameStart := 0 },
  { event := event107084
    frameStart := 0 },
  { event := event107085
    frameStart := 0 },
  { event := event107086
    frameStart := 0 },
  { event := event107087
    frameStart := 0 }
]

def eventLeaf6693 : Array AnnotatedEvent := #[
  { event := event107088
    frameStart := 0 },
  { event := event107089
    frameStart := 0 },
  { event := event107090
    frameStart := 0 },
  { event := event107091
    frameStart := 0 },
  { event := event107092
    frameStart := 0 },
  { event := event107093
    frameStart := 0 },
  { event := event107094
    frameStart := 0 },
  { event := event107095
    frameStart := 0 },
  { event := event107096
    frameStart := 0 },
  { event := event107097
    frameStart := 0 },
  { event := event107098
    frameStart := 0 },
  { event := event107099
    frameStart := 0 },
  { event := event107100
    frameStart := 0 },
  { event := event107101
    frameStart := 0 },
  { event := event107102
    frameStart := 0 },
  { event := event107103
    frameStart := 0 }
]

def eventLeaf6694 : Array AnnotatedEvent := #[
  { event := event107104
    frameStart := 0 },
  { event := event107105
    frameStart := 0 },
  { event := event107106
    frameStart := 0 },
  { event := event107107
    frameStart := 0 },
  { event := event107108
    frameStart := 0 },
  { event := event107109
    frameStart := 0 },
  { event := event107110
    frameStart := 0 },
  { event := event107111
    frameStart := 0 },
  { event := event107112
    frameStart := 0 },
  { event := event107113
    frameStart := 0 },
  { event := event107114
    frameStart := 0 },
  { event := event107115
    frameStart := 0 },
  { event := event107116
    frameStart := 0 },
  { event := event107117
    frameStart := 0 },
  { event := event107118
    frameStart := 0 },
  { event := event107119
    frameStart := 0 }
]

def eventLeaf6695 : Array AnnotatedEvent := #[
  { event := event107120
    frameStart := 0 },
  { event := event107121
    frameStart := 0 },
  { event := event107122
    frameStart := 0 },
  { event := event107123
    frameStart := 0 },
  { event := event107124
    frameStart := 0 },
  { event := event107125
    frameStart := 0 },
  { event := event107126
    frameStart := 0 },
  { event := event107127
    frameStart := 0 },
  { event := event107128
    frameStart := 0 },
  { event := event107129
    frameStart := 0 },
  { event := event107130
    frameStart := 0 },
  { event := event107131
    frameStart := 0 },
  { event := event107132
    frameStart := 0 },
  { event := event107133
    frameStart := 0 },
  { event := event107134
    frameStart := 0 },
  { event := event107135
    frameStart := 0 }
]

def eventLeaf6696 : Array AnnotatedEvent := #[
  { event := event107136
    frameStart := 0 },
  { event := event107137
    frameStart := 0 },
  { event := event107138
    frameStart := 0 },
  { event := event107139
    frameStart := 0 },
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
    frameStart := 107180 },
  { event := event107181
    frameStart := 107180 },
  { event := event107182
    frameStart := 107180 },
  { event := event107183
    frameStart := 107180 }
]

def eventLeaf6699 : Array AnnotatedEvent := #[
  { event := event107184
    frameStart := 107180 },
  { event := event107185
    frameStart := 107180 },
  { event := event107186
    frameStart := 107180 },
  { event := event107187
    frameStart := 107180 },
  { event := event107188
    frameStart := 107180 },
  { event := event107189
    frameStart := 107180 },
  { event := event107190
    frameStart := 107180 },
  { event := event107191
    frameStart := 107180 },
  { event := event107192
    frameStart := 107180 },
  { event := event107193
    frameStart := 107180 },
  { event := event107194
    frameStart := 107180 },
  { event := event107195
    frameStart := 107180 },
  { event := event107196
    frameStart := 107180 },
  { event := event107197
    frameStart := 107180 },
  { event := event107198
    frameStart := 107180 },
  { event := event107199
    frameStart := 107180 }
]

def eventLeaf6700 : Array AnnotatedEvent := #[
  { event := event107200
    frameStart := 107180 },
  { event := event107201
    frameStart := 107180 },
  { event := event107202
    frameStart := 107180 },
  { event := event107203
    frameStart := 107180 },
  { event := event107204
    frameStart := 107180 },
  { event := event107205
    frameStart := 107180 },
  { event := event107206
    frameStart := 107180 },
  { event := event107207
    frameStart := 107180 },
  { event := event107208
    frameStart := 107180 },
  { event := event107209
    frameStart := 107180 },
  { event := event107210
    frameStart := 107180 },
  { event := event107211
    frameStart := 107180 },
  { event := event107212
    frameStart := 107180 },
  { event := event107213
    frameStart := 107180 },
  { event := event107214
    frameStart := 107180 },
  { event := event107215
    frameStart := 107180 }
]

def eventLeaf6701 : Array AnnotatedEvent := #[
  { event := event107216
    frameStart := 107180 },
  { event := event107217
    frameStart := 107180 },
  { event := event107218
    frameStart := 107180 },
  { event := event107219
    frameStart := 107180 },
  { event := event107220
    frameStart := 107180 },
  { event := event107221
    frameStart := 107180 },
  { event := event107222
    frameStart := 107180 },
  { event := event107223
    frameStart := 107180 },
  { event := event107224
    frameStart := 107180 },
  { event := event107225
    frameStart := 107180 },
  { event := event107226
    frameStart := 107180 },
  { event := event107227
    frameStart := 107180 },
  { event := event107228
    frameStart := 107228 },
  { event := event107229
    frameStart := 107228 },
  { event := event107230
    frameStart := 107228 },
  { event := event107231
    frameStart := 107228 }
]

def eventLeaf6702 : Array AnnotatedEvent := #[
  { event := event107232
    frameStart := 107228 },
  { event := event107233
    frameStart := 107228 },
  { event := event107234
    frameStart := 107228 },
  { event := event107235
    frameStart := 107228 },
  { event := event107236
    frameStart := 107228 },
  { event := event107237
    frameStart := 107228 },
  { event := event107238
    frameStart := 107228 },
  { event := event107239
    frameStart := 107228 },
  { event := event107240
    frameStart := 107228 },
  { event := event107241
    frameStart := 107228 },
  { event := event107242
    frameStart := 107228 },
  { event := event107243
    frameStart := 107228 },
  { event := event107244
    frameStart := 107228 },
  { event := event107245
    frameStart := 107228 },
  { event := event107246
    frameStart := 107228 },
  { event := event107247
    frameStart := 107228 }
]

def eventLeaf6703 : Array AnnotatedEvent := #[
  { event := event107248
    frameStart := 107228 },
  { event := event107249
    frameStart := 107228 },
  { event := event107250
    frameStart := 107228 },
  { event := event107251
    frameStart := 107228 },
  { event := event107252
    frameStart := 107228 },
  { event := event107253
    frameStart := 107228 },
  { event := event107254
    frameStart := 107228 },
  { event := event107255
    frameStart := 107228 },
  { event := event107256
    frameStart := 107228 },
  { event := event107257
    frameStart := 107228 },
  { event := event107258
    frameStart := 107228 },
  { event := event107259
    frameStart := 107228 },
  { event := event107260
    frameStart := 107228 },
  { event := event107261
    frameStart := 107228 },
  { event := event107262
    frameStart := 107228 },
  { event := event107263
    frameStart := 107228 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events418
