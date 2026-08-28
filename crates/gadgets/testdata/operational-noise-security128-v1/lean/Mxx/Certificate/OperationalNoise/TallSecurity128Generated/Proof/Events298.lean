import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events298

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event76288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47979⟩⟩) (.product (.predecessor 0 76286 .coefficient) (.predecessor 1 76287 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47979⟩⟩, .operator (⟨76285, 0⟩, ⟨76282, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩)

def exact76290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩]

theorem exact76290RawTermsValid :
    exact76290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47979⟩⟩) exact76290RawTerms (.finite 3600) 76288 .exactZero (none)

def event76291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47980⟩⟩) 0 ⟨47979⟩ 76290

def event76292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47980⟩⟩) (.identity (.predecessor 0 76291 .coefficient))

def event76293 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47980⟩⟩) (.finite 3600)

def event76294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48196⟩⟩) 0 ⟨47980⟩ 76293

def event76295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48196⟩⟩) (.authority (.programFamilyFact))

def exact76296RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], []⟩, (1)⟩]

theorem exact76296RawTermsValid :
    exact76296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48196⟩⟩) exact76296RawTerms (.finite 60) 76295 .exactZero (none)

def event76297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48197⟩⟩) 0 ⟨48196⟩ 76296

def event76298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48197⟩⟩) (.identity (.predecessor 0 76297 .coefficient))

def event76299 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48197⟩⟩) (.finite 60)

def event76300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49353⟩⟩) 0 ⟨48197⟩ 76299

def event76301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.authority (.programFamilyFact))

def event76302 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.finite 3720)

def event76303 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event76304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49355⟩⟩) 0 ⟨7177⟩ 76303

def event76305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49355⟩⟩) 1 ⟨49353⟩ 76302

def event76306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49355⟩⟩) (.authority (.operator))

def exact76307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩]

theorem exact76307RawTermsValid :
    exact76307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49355⟩⟩) exact76307RawTerms .large 76306 .exactZero (none)

def event76308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50179⟩⟩) 0 ⟨49355⟩ 76307

def event76309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50179⟩⟩) (.authority (.operator))

def exact76310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (1)⟩]

theorem exact76310RawTermsValid :
    exact76310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50179⟩⟩) exact76310RawTerms (.finite 8192) 76309 .exactZero (none)

def event76311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event76312 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event76313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49530⟩⟩) 0 ⟨48197⟩ 76299

def event76314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49530⟩⟩) 1 ⟨136⟩ 76312

def event76315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49530⟩⟩) (.sum [.predecessor 0 76313 .coefficient, .predecessor 1 76314 .coefficient])

def event76316 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49530⟩⟩) (.finite 60)

def event76317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49531⟩⟩) 0 ⟨49530⟩ 76316

def event76318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49531⟩⟩) (.identity (.predecessor 0 76317 .coefficient))

def exact76319RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], []⟩, (1)⟩]

theorem exact76319RawTermsValid :
    exact76319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76319 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49531⟩⟩) exact76319RawTerms (.finite 60) 76318 .exactZero (none)

def event76320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact76321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76321RawTermsValid :
    exact76321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact76321RawTerms .large 76320 .exactZero (none)

def event76322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49532⟩⟩) 0 ⟨6908⟩ 76321

def event76323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49532⟩⟩) 1 ⟨49531⟩ 76319

def event76324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49532⟩⟩) (.product (.predecessor 0 76322 .coefficient) (.predecessor 1 76323 .coefficient) (⟨false, false, none, none, none⟩))

def event76325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49532⟩⟩, .operator (⟨76321, 0⟩, ⟨76319, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76326RawTermsValid :
    exact76326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49532⟩⟩) exact76326RawTerms .large 76324 .exactZero (none)

def event76327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 76303

def event76328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact76329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact76329RawTermsValid :
    exact76329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact76329RawTerms .large 76328 .exactZero (none)

def event76330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49533⟩⟩) 0 ⟨7196⟩ 76329

def event76331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49533⟩⟩) 1 ⟨49532⟩ 76326

def event76332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49533⟩⟩) (.sum [.predecessor 0 76330 .coefficient, .predecessor 1 76331 .coefficient])

def exact76333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76333RawTermsValid :
    exact76333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49533⟩⟩) exact76333RawTerms .large 76332 .exactZero (none)

def event76334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50180⟩⟩) 0 ⟨49533⟩ 76333

def event76335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50180⟩⟩) 1 ⟨50179⟩ 76310

def event76336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50180⟩⟩) (.product (.predecessor 0 76334 .coefficient) (.predecessor 1 76335 .coefficient) (⟨false, false, none, none, none⟩))

def event76337 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50180⟩⟩, .operator (⟨76333, 0⟩, ⟨76310, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (1)⟩)

def event76338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50180⟩⟩, .operator (⟨76333, 1⟩, ⟨76310, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩)

def event76339 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50180⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50179⟩⟩) ⟨49355⟩ 76307)

def event76340 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50180⟩⟩, .relation 76339 0, ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (-1)⟩)

def exact76341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (-1)⟩]

theorem exact76341RawTermsValid :
    exact76341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50180⟩⟩) exact76341RawTerms .large 76336 .exactZero (none)

def event76342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48441⟩⟩) 0 ⟨48197⟩ 76299

def event76343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48441⟩⟩) (.authority (.programFamilyFact))

def exact76344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], []⟩, (1)⟩]

theorem exact76344RawTermsValid :
    exact76344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48441⟩⟩) exact76344RawTerms (.finite 63) 76343 .exactZero (none)

def event76345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48442⟩⟩) 0 ⟨6908⟩ 76321

def event76346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48442⟩⟩) 1 ⟨48441⟩ 76344

def event76347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48442⟩⟩) (.product (.predecessor 0 76345 .coefficient) (.predecessor 1 76346 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76348 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48442⟩⟩, .operator (⟨76321, 0⟩, ⟨76344, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76349RawTermsValid :
    exact76349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48442⟩⟩) exact76349RawTerms .large 76347 .exactZero (none)

def event76350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 76303

def event76351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact76352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact76352RawTermsValid :
    exact76352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact76352RawTerms .large 76351 .exactZero (none)

def event76353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48443⟩⟩) 0 ⟨7232⟩ 76352

def event76354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48443⟩⟩) 1 ⟨48442⟩ 76349

def event76355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48443⟩⟩) (.sum [.predecessor 0 76353 .coefficient, .predecessor 1 76354 .coefficient])

def exact76356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76356RawTermsValid :
    exact76356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48443⟩⟩) exact76356RawTerms .large 76355 .exactZero (none)

def event76357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50183⟩⟩) 0 ⟨48443⟩ 76356

def event76358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50183⟩⟩) 1 ⟨50180⟩ 76341

def event76359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50183⟩⟩) (.sum [.predecessor 0 76357 .coefficient, .predecessor 1 76358 .coefficient])

def exact76360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76360RawTermsValid :
    exact76360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50183⟩⟩) exact76360RawTerms .large 76359 .exactZero (none)

def event76361 : Event := .preFoldPolynomial 76360 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact76362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event76362 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50183⟩⟩) 76361 exact76362RawTerms .large 76359 .exactZero (none)

def event76363 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48197⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨76205, 76363⟩

def event76364 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49016⟩⟩]⟩) (1) 0 2 (.universal 76363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49016⟩⟩]⟩) (none) 76362)

def event76365 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49019⟩⟩, .relation 76364 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def event76366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49019⟩⟩, .relation 76364 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩)

def event76367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49019⟩⟩, .relation 76364 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩)

def event76368 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49019⟩⟩, .relation 76364 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact76369RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76369RawTermsValid :
    exact76369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49019⟩⟩) exact76369RawTerms .large 76201 (.finite 202072841853861888) (some (76203))

def event76370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50182⟩⟩) 0 ⟨49019⟩ 76369

def event76371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50182⟩⟩) 1 ⟨50181⟩ 76191

def event76372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50182⟩⟩) (.sum [.predecessor 0 76370 .coefficient, .predecessor 1 76371 .coefficient])

def event76373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50182⟩⟩, .operator (⟨76369, 0⟩, ⟨76191, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (1)⟩)

def event76374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50182⟩⟩, .operator (⟨76369, 2⟩, ⟨76191, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (-1)⟩)

def event76375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50182⟩⟩) (.sum [.result 76369 .summary, .result 76191 .summary])

def exact76376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76376RawTermsValid :
    exact76376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50182⟩⟩) exact76376RawTerms .large 76372 (.finite 32194504275408640829496428331008) (some (76375))

def event76377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46673⟩⟩) 0 ⟨45517⟩ 3126

def event76378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.authority (.programFamilyFact))

def event76379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.finite 3720)

def event76380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46675⟩⟩) 0 ⟨7177⟩ 15500

def event76381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46675⟩⟩) 1 ⟨46673⟩ 76379

def event76382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46675⟩⟩) (.authority (.operator))

def exact76383RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46675⟩⟩]⟩, (1)⟩]

theorem exact76383RawTermsValid :
    exact76383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76383 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46675⟩⟩) exact76383RawTerms .large 76382 .exactZero (none)

def event76384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47499⟩⟩) 0 ⟨46675⟩ 76383

def event76385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47499⟩⟩) (.authority (.operator))

def exact76386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩, (1)⟩]

theorem exact76386RawTermsValid :
    exact76386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47499⟩⟩) exact76386RawTerms (.finite 8192) 76385 .exactZero (none)

def event76387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46504⟩⟩) 0 ⟨45300⟩ 3120

def event76388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46504⟩⟩) (.authority (.programFamilyFact))

def event76389 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46504⟩⟩) (.finite 3720)

def event76390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46505⟩⟩) 0 ⟨7177⟩ 15500

def event76391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46505⟩⟩) 1 ⟨46504⟩ 76389

def event76392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46505⟩⟩) (.authority (.operator))

def exact76393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩]

theorem exact76393RawTermsValid :
    exact76393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46505⟩⟩) exact76393RawTerms .large 76392 .exactZero (none)

def event76394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47045⟩⟩) 0 ⟨46505⟩ 76393

def event76395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47045⟩⟩) (.authority (.operator))

def exact76396RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩]

theorem exact76396RawTermsValid :
    exact76396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47045⟩⟩) exact76396RawTerms (.finite 8192) 76395 .exactZero (none)

def event76397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45301⟩⟩) 0 ⟨45298⟩ 3109

def event76398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45301⟩⟩) 1 ⟨10328⟩ 75903

def event76399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45301⟩⟩) (.tensor (.predecessor 0 76397 .coefficient) (.predecessor 1 76398 .coefficient) true false)

def event76400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45301⟩⟩, .operator (⟨3109, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76401RawTermsValid :
    exact76401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45301⟩⟩) exact76401RawTerms .large 76399 .exactZero (none)

def event76402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10342⟩⟩) 0 ⟨10327⟩ 75773

def event76403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10342⟩⟩) 1 ⟨7284⟩ 17581

def event76404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10342⟩⟩) (.product (.predecessor 0 76402 .coefficient) (.predecessor 1 76403 .coefficient) (⟨false, false, none, none, none⟩))

def event76405 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10342⟩⟩, .operator (⟨75773, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact76406RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact76406RawTermsValid :
    exact76406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76406 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10342⟩⟩) exact76406RawTerms .large 76404 .exactZero (none)

def event76407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45302⟩⟩) 0 ⟨10342⟩ 76406

def event76408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45302⟩⟩) 1 ⟨45301⟩ 76401

def event76409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45302⟩⟩) (.sum [.predecessor 0 76407 .coefficient, .predecessor 1 76408 .coefficient])

def exact76410RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76410RawTermsValid :
    exact76410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76410 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45302⟩⟩) exact76410RawTerms .large 76409 .exactZero (none)

def event76411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45303⟩⟩) 0 ⟨45302⟩ 76410

def event76412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45303⟩⟩) 1 ⟨110⟩ 17573

def event76413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45303⟩⟩) (.sum [.predecessor 0 76411 .coefficient, .predecessor 1 76412 .coefficient])

def event76414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45303⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩) [⟨.result 17573 .coefficient, false, none⟩])

def event76415 : Event := .survivorFold (1) 76414

def exact76416RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76416RawTermsValid :
    exact76416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45303⟩⟩) exact76416RawTerms .large 76413 (.finite 26) (some (76414))

def event76417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45304⟩⟩) 0 ⟨45303⟩ 76416

def event76418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45304⟩⟩) 1 ⟨14871⟩ 3112

def event76419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45304⟩⟩) (.product (.predecessor 0 76417 .coefficient) (.predecessor 1 76418 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45304⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩) [⟨.result 3112 .coefficient, true, some 1⟩])

def event76421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45304⟩⟩) (.product (.result 76416 .summary) (.transfer 76420) (⟨false, false, none, none, none⟩))

def event76422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45304⟩⟩, .operator (⟨76416, 1⟩, ⟨3112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event76423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45304⟩⟩, .operator (⟨76416, 0⟩, ⟨3112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact76424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76424RawTermsValid :
    exact76424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45304⟩⟩) exact76424RawTerms .large 76419 (.finite 49414144) (some (76421))

def event76425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14872⟩⟩) 0 ⟨14871⟩ 3112

def event76426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14872⟩⟩) 1 ⟨10328⟩ 75903

def event76427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14872⟩⟩) (.tensor (.predecessor 0 76425 .coefficient) (.predecessor 1 76426 .coefficient) true false)

def event76428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14872⟩⟩, .operator (⟨3112, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76429RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76429RawTermsValid :
    exact76429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76429 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14872⟩⟩) exact76429RawTerms .large 76427 .exactZero (none)

def event76430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10359⟩⟩) 0 ⟨10327⟩ 75773

def event76431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10359⟩⟩) 1 ⟨7301⟩ 17622

def event76432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10359⟩⟩) (.product (.predecessor 0 76430 .coefficient) (.predecessor 1 76431 .coefficient) (⟨false, false, none, none, none⟩))

def event76433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10359⟩⟩, .operator (⟨75773, 0⟩, ⟨17622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩)

def exact76434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact76434RawTermsValid :
    exact76434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10359⟩⟩) exact76434RawTerms .large 76432 .exactZero (none)

def event76435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14873⟩⟩) 0 ⟨10359⟩ 76434

def event76436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14873⟩⟩) 1 ⟨14872⟩ 76429

def event76437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14873⟩⟩) (.sum [.predecessor 0 76435 .coefficient, .predecessor 1 76436 .coefficient])

def exact76438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76438RawTermsValid :
    exact76438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14873⟩⟩) exact76438RawTerms .large 76437 .exactZero (none)

def event76439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14874⟩⟩) 0 ⟨14873⟩ 76438

def event76440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14874⟩⟩) 1 ⟨127⟩ 17614

def event76441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14874⟩⟩) (.sum [.predecessor 0 76439 .coefficient, .predecessor 1 76440 .coefficient])

def event76442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14874⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩) [⟨.result 17614 .coefficient, false, none⟩])

def event76443 : Event := .survivorFold (1) 76442

def exact76444RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76444RawTermsValid :
    exact76444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14874⟩⟩) exact76444RawTerms .large 76441 (.finite 26) (some (76442))

def event76445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14875⟩⟩) 0 ⟨14874⟩ 76444

def event76446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14875⟩⟩) 1 ⟨9563⟩ 17611

def event76447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14875⟩⟩) (.product (.predecessor 0 76445 .coefficient) (.predecessor 1 76446 .coefficient) (⟨false, false, none, none, none⟩))

def event76448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14875⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) [⟨.result 17607 .coefficient, false, none⟩])

def event76449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14875⟩⟩) (.product (.result 76444 .summary) (.transfer 76448) (⟨false, false, none, none, none⟩))

def event76450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14875⟩⟩, .operator (⟨76444, 1⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (-1)⟩)

def event76451 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14875⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581)

def event76452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14875⟩⟩, .relation 76451 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩)

def event76453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14875⟩⟩, .operator (⟨76444, 0⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact76454RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩]

theorem exact76454RawTermsValid :
    exact76454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14875⟩⟩) exact76454RawTerms .large 76447 (.finite 279172874240) (some (76449))

def event76455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45305⟩⟩) 0 ⟨14875⟩ 76454

def event76456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45305⟩⟩) 1 ⟨45304⟩ 76424

def event76457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45305⟩⟩) (.sum [.predecessor 0 76455 .coefficient, .predecessor 1 76456 .coefficient])

def event76458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45305⟩⟩, .operator (⟨76454, 1⟩, ⟨76424, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def event76459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45305⟩⟩) (.sum [.result 76454 .summary, .result 76424 .summary])

def exact76460RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76460RawTermsValid :
    exact76460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45305⟩⟩) exact76460RawTerms .large 76457 (.finite 279222288384) (some (76459))

def event76461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47046⟩⟩) 0 ⟨45305⟩ 76460

def event76462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47046⟩⟩) 1 ⟨47045⟩ 76396

def event76463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47046⟩⟩) (.product (.predecessor 0 76461 .coefficient) (.predecessor 1 76462 .coefficient) (⟨false, false, none, none, none⟩))

def event76464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47046⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩) [⟨.result 76396 .coefficient, false, none⟩])

def event76465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47046⟩⟩) (.product (.result 76460 .summary) (.transfer 76464) (⟨false, false, none, none, none⟩))

def event76466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47046⟩⟩, .operator (⟨76460, 1⟩, ⟨76396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩)

def event76467 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47046⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47045⟩⟩) ⟨46505⟩ 76393)

def event76468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47046⟩⟩, .relation 76467 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (-1)⟩)

def event76469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47046⟩⟩, .operator (⟨76460, 0⟩, ⟨76396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩)

def exact76470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (-1)⟩]

theorem exact76470RawTermsValid :
    exact76470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47046⟩⟩) exact76470RawTerms .large 76463 (.finite 2998126492308901724160) (some (76465))

def event76471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45969⟩⟩) 0 ⟨45300⟩ 3120

def event76472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45969⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact76473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩]

theorem exact76473RawTermsValid :
    exact76473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45969⟩⟩) exact76473RawTerms (.finite 5647228698) 76472 .exactZero (none)

def event76474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45971⟩⟩) 0 ⟨45969⟩ 76473

def event76475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45971⟩⟩) 1 ⟨2370⟩ 4

def event76476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45971⟩⟩) (.scale (.predecessor 0 76474 .coefficient) (.value (.predecessor 1 76475 .coefficient)))

def exact76477RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩]

theorem exact76477RawTermsValid :
    exact76477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76477 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45971⟩⟩) exact76477RawTerms (.finite 5647228698) 76476 .exactZero (none)

def event76478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45972⟩⟩) 0 ⟨10368⟩ 75995

def event76479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45972⟩⟩) 1 ⟨45971⟩ 76477

def event76480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45972⟩⟩) (.product (.predecessor 0 76478 .coefficient) (.predecessor 1 76479 .coefficient) (⟨false, false, none, none, none⟩))

def event76481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45972⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩) [⟨.result 76473 .coefficient, false, none⟩])

def event76482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45972⟩⟩) (.product (.result 75995 .summary) (.transfer 76481) (⟨false, false, none, none, none⟩))

def event76483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45972⟩⟩, .operator (⟨75995, 0⟩, ⟨76477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩)

def event76484 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨45970⟩⟩)

def event76485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event76486 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event76487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event76488 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event76489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event76490 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event76491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event76492 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event76493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 76492

def event76494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 76490

def event76495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 76493 .coefficient) (.value (.predecessor 1 76494 .coefficient)))

def event76496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event76497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 76496

def event76498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 76488

def event76499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 76497 .coefficient, .predecessor 1 76498 .coefficient])

def event76500 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event76501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 76500

def event76502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 76486

def event76503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 76502 .coefficient))

def event76504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event76505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 76504

def event76506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact76507RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76507RawTermsValid :
    exact76507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact76507RawTerms (.finite 58) 76506 .exactZero (none)

def event76508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 76504

def event76509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact76510RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact76510RawTermsValid :
    exact76510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact76510RawTerms (.finite 58) 76509 .exactZero (none)

def event76511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 76510

def event76512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 76507

def event76513 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 76511 .coefficient) (.predecessor 1 76512 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩) [⟨.result 76510 .coefficient, true, some 1⟩, ⟨.result 76507 .coefficient, true, some 1⟩])

def event76515 : Event := .survivorFold (1) 76514

def exact76516RawTerms : List Term := []

theorem exact76516RawTermsValid :
    exact76516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact76516RawTerms (.finite 3364) 76513 (.finite 3364) (some (76514))

def event76517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 76516

def event76518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 76517 .coefficient))

def event76519 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event76520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45969⟩⟩) 0 ⟨45300⟩ 76519

def event76521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45969⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact76522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩]

theorem exact76522RawTermsValid :
    exact76522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45969⟩⟩) exact76522RawTerms (.finite 5647228698) 76521 .exactZero (none)

def event76523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact76524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact76524RawTermsValid :
    exact76524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact76524RawTerms .large 76523 .exactZero (none)

def event76525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45970⟩⟩) 0 ⟨35⟩ 76524

def event76526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45970⟩⟩) 1 ⟨45969⟩ 76522

def event76527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45970⟩⟩) (.product (.predecessor 0 76525 .coefficient) (.predecessor 1 76526 .coefficient) (⟨false, false, none, none, none⟩))

def event76528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45970⟩⟩, .operator (⟨76524, 0⟩, ⟨76522, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩)

def exact76529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩]

theorem exact76529RawTermsValid :
    exact76529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45970⟩⟩) exact76529RawTerms .large 76527 .exactZero (none)

def event76530 : Event := .preFoldPolynomial 76529 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩] .exactZero none

def exact76531RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩, (1)⟩]

def event76531 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨45970⟩⟩) 76530 exact76531RawTerms .large 76527 .exactZero (none)

def event76532 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47049⟩⟩)

def event76533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event76534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event76535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event76536 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event76537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event76538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event76539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event76540 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event76541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 76540

def event76542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 76538

def event76543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 76541 .coefficient) (.value (.predecessor 1 76542 .coefficient)))

def eventLeaf4768 : Array AnnotatedEvent := #[
  { event := event76288
    frameStart := 76259 },
  { event := event76289
    frameStart := 76259 },
  { event := event76290
    frameStart := 76259 },
  { event := event76291
    frameStart := 76259 },
  { event := event76292
    frameStart := 76259 },
  { event := event76293
    frameStart := 76259 },
  { event := event76294
    frameStart := 76259 },
  { event := event76295
    frameStart := 76259 },
  { event := event76296
    frameStart := 76259 },
  { event := event76297
    frameStart := 76259 },
  { event := event76298
    frameStart := 76259 },
  { event := event76299
    frameStart := 76259 },
  { event := event76300
    frameStart := 76259 },
  { event := event76301
    frameStart := 76259 },
  { event := event76302
    frameStart := 76259 },
  { event := event76303
    frameStart := 76259 }
]

def eventLeaf4769 : Array AnnotatedEvent := #[
  { event := event76304
    frameStart := 76259 },
  { event := event76305
    frameStart := 76259 },
  { event := event76306
    frameStart := 76259 },
  { event := event76307
    frameStart := 76259 },
  { event := event76308
    frameStart := 76259 },
  { event := event76309
    frameStart := 76259 },
  { event := event76310
    frameStart := 76259 },
  { event := event76311
    frameStart := 76259 },
  { event := event76312
    frameStart := 76259 },
  { event := event76313
    frameStart := 76259 },
  { event := event76314
    frameStart := 76259 },
  { event := event76315
    frameStart := 76259 },
  { event := event76316
    frameStart := 76259 },
  { event := event76317
    frameStart := 76259 },
  { event := event76318
    frameStart := 76259 },
  { event := event76319
    frameStart := 76259 }
]

def eventLeaf4770 : Array AnnotatedEvent := #[
  { event := event76320
    frameStart := 76259 },
  { event := event76321
    frameStart := 76259 },
  { event := event76322
    frameStart := 76259 },
  { event := event76323
    frameStart := 76259 },
  { event := event76324
    frameStart := 76259 },
  { event := event76325
    frameStart := 76259 },
  { event := event76326
    frameStart := 76259 },
  { event := event76327
    frameStart := 76259 },
  { event := event76328
    frameStart := 76259 },
  { event := event76329
    frameStart := 76259 },
  { event := event76330
    frameStart := 76259 },
  { event := event76331
    frameStart := 76259 },
  { event := event76332
    frameStart := 76259 },
  { event := event76333
    frameStart := 76259 },
  { event := event76334
    frameStart := 76259 },
  { event := event76335
    frameStart := 76259 }
]

def eventLeaf4771 : Array AnnotatedEvent := #[
  { event := event76336
    frameStart := 76259 },
  { event := event76337
    frameStart := 76259 },
  { event := event76338
    frameStart := 76259 },
  { event := event76339
    frameStart := 76259 },
  { event := event76340
    frameStart := 76259 },
  { event := event76341
    frameStart := 76259 },
  { event := event76342
    frameStart := 76259 },
  { event := event76343
    frameStart := 76259 },
  { event := event76344
    frameStart := 76259 },
  { event := event76345
    frameStart := 76259 },
  { event := event76346
    frameStart := 76259 },
  { event := event76347
    frameStart := 76259 },
  { event := event76348
    frameStart := 76259 },
  { event := event76349
    frameStart := 76259 },
  { event := event76350
    frameStart := 76259 },
  { event := event76351
    frameStart := 76259 }
]

def eventLeaf4772 : Array AnnotatedEvent := #[
  { event := event76352
    frameStart := 76259 },
  { event := event76353
    frameStart := 76259 },
  { event := event76354
    frameStart := 76259 },
  { event := event76355
    frameStart := 76259 },
  { event := event76356
    frameStart := 76259 },
  { event := event76357
    frameStart := 76259 },
  { event := event76358
    frameStart := 76259 },
  { event := event76359
    frameStart := 76259 },
  { event := event76360
    frameStart := 76259 },
  { event := event76361
    frameStart := 76259 },
  { event := event76362
    frameStart := 76259 },
  { event := event76363
    frameStart := 0 },
  { event := event76364
    frameStart := 0 },
  { event := event76365
    frameStart := 0 },
  { event := event76366
    frameStart := 0 },
  { event := event76367
    frameStart := 0 }
]

def eventLeaf4773 : Array AnnotatedEvent := #[
  { event := event76368
    frameStart := 0 },
  { event := event76369
    frameStart := 0 },
  { event := event76370
    frameStart := 0 },
  { event := event76371
    frameStart := 0 },
  { event := event76372
    frameStart := 0 },
  { event := event76373
    frameStart := 0 },
  { event := event76374
    frameStart := 0 },
  { event := event76375
    frameStart := 0 },
  { event := event76376
    frameStart := 0 },
  { event := event76377
    frameStart := 0 },
  { event := event76378
    frameStart := 0 },
  { event := event76379
    frameStart := 0 },
  { event := event76380
    frameStart := 0 },
  { event := event76381
    frameStart := 0 },
  { event := event76382
    frameStart := 0 },
  { event := event76383
    frameStart := 0 }
]

def eventLeaf4774 : Array AnnotatedEvent := #[
  { event := event76384
    frameStart := 0 },
  { event := event76385
    frameStart := 0 },
  { event := event76386
    frameStart := 0 },
  { event := event76387
    frameStart := 0 },
  { event := event76388
    frameStart := 0 },
  { event := event76389
    frameStart := 0 },
  { event := event76390
    frameStart := 0 },
  { event := event76391
    frameStart := 0 },
  { event := event76392
    frameStart := 0 },
  { event := event76393
    frameStart := 0 },
  { event := event76394
    frameStart := 0 },
  { event := event76395
    frameStart := 0 },
  { event := event76396
    frameStart := 0 },
  { event := event76397
    frameStart := 0 },
  { event := event76398
    frameStart := 0 },
  { event := event76399
    frameStart := 0 }
]

def eventLeaf4775 : Array AnnotatedEvent := #[
  { event := event76400
    frameStart := 0 },
  { event := event76401
    frameStart := 0 },
  { event := event76402
    frameStart := 0 },
  { event := event76403
    frameStart := 0 },
  { event := event76404
    frameStart := 0 },
  { event := event76405
    frameStart := 0 },
  { event := event76406
    frameStart := 0 },
  { event := event76407
    frameStart := 0 },
  { event := event76408
    frameStart := 0 },
  { event := event76409
    frameStart := 0 },
  { event := event76410
    frameStart := 0 },
  { event := event76411
    frameStart := 0 },
  { event := event76412
    frameStart := 0 },
  { event := event76413
    frameStart := 0 },
  { event := event76414
    frameStart := 0 },
  { event := event76415
    frameStart := 0 }
]

def eventLeaf4776 : Array AnnotatedEvent := #[
  { event := event76416
    frameStart := 0 },
  { event := event76417
    frameStart := 0 },
  { event := event76418
    frameStart := 0 },
  { event := event76419
    frameStart := 0 },
  { event := event76420
    frameStart := 0 },
  { event := event76421
    frameStart := 0 },
  { event := event76422
    frameStart := 0 },
  { event := event76423
    frameStart := 0 },
  { event := event76424
    frameStart := 0 },
  { event := event76425
    frameStart := 0 },
  { event := event76426
    frameStart := 0 },
  { event := event76427
    frameStart := 0 },
  { event := event76428
    frameStart := 0 },
  { event := event76429
    frameStart := 0 },
  { event := event76430
    frameStart := 0 },
  { event := event76431
    frameStart := 0 }
]

def eventLeaf4777 : Array AnnotatedEvent := #[
  { event := event76432
    frameStart := 0 },
  { event := event76433
    frameStart := 0 },
  { event := event76434
    frameStart := 0 },
  { event := event76435
    frameStart := 0 },
  { event := event76436
    frameStart := 0 },
  { event := event76437
    frameStart := 0 },
  { event := event76438
    frameStart := 0 },
  { event := event76439
    frameStart := 0 },
  { event := event76440
    frameStart := 0 },
  { event := event76441
    frameStart := 0 },
  { event := event76442
    frameStart := 0 },
  { event := event76443
    frameStart := 0 },
  { event := event76444
    frameStart := 0 },
  { event := event76445
    frameStart := 0 },
  { event := event76446
    frameStart := 0 },
  { event := event76447
    frameStart := 0 }
]

def eventLeaf4778 : Array AnnotatedEvent := #[
  { event := event76448
    frameStart := 0 },
  { event := event76449
    frameStart := 0 },
  { event := event76450
    frameStart := 0 },
  { event := event76451
    frameStart := 0 },
  { event := event76452
    frameStart := 0 },
  { event := event76453
    frameStart := 0 },
  { event := event76454
    frameStart := 0 },
  { event := event76455
    frameStart := 0 },
  { event := event76456
    frameStart := 0 },
  { event := event76457
    frameStart := 0 },
  { event := event76458
    frameStart := 0 },
  { event := event76459
    frameStart := 0 },
  { event := event76460
    frameStart := 0 },
  { event := event76461
    frameStart := 0 },
  { event := event76462
    frameStart := 0 },
  { event := event76463
    frameStart := 0 }
]

def eventLeaf4779 : Array AnnotatedEvent := #[
  { event := event76464
    frameStart := 0 },
  { event := event76465
    frameStart := 0 },
  { event := event76466
    frameStart := 0 },
  { event := event76467
    frameStart := 0 },
  { event := event76468
    frameStart := 0 },
  { event := event76469
    frameStart := 0 },
  { event := event76470
    frameStart := 0 },
  { event := event76471
    frameStart := 0 },
  { event := event76472
    frameStart := 0 },
  { event := event76473
    frameStart := 0 },
  { event := event76474
    frameStart := 0 },
  { event := event76475
    frameStart := 0 },
  { event := event76476
    frameStart := 0 },
  { event := event76477
    frameStart := 0 },
  { event := event76478
    frameStart := 0 },
  { event := event76479
    frameStart := 0 }
]

def eventLeaf4780 : Array AnnotatedEvent := #[
  { event := event76480
    frameStart := 0 },
  { event := event76481
    frameStart := 0 },
  { event := event76482
    frameStart := 0 },
  { event := event76483
    frameStart := 0 },
  { event := event76484
    frameStart := 76484 },
  { event := event76485
    frameStart := 76484 },
  { event := event76486
    frameStart := 76484 },
  { event := event76487
    frameStart := 76484 },
  { event := event76488
    frameStart := 76484 },
  { event := event76489
    frameStart := 76484 },
  { event := event76490
    frameStart := 76484 },
  { event := event76491
    frameStart := 76484 },
  { event := event76492
    frameStart := 76484 },
  { event := event76493
    frameStart := 76484 },
  { event := event76494
    frameStart := 76484 },
  { event := event76495
    frameStart := 76484 }
]

def eventLeaf4781 : Array AnnotatedEvent := #[
  { event := event76496
    frameStart := 76484 },
  { event := event76497
    frameStart := 76484 },
  { event := event76498
    frameStart := 76484 },
  { event := event76499
    frameStart := 76484 },
  { event := event76500
    frameStart := 76484 },
  { event := event76501
    frameStart := 76484 },
  { event := event76502
    frameStart := 76484 },
  { event := event76503
    frameStart := 76484 },
  { event := event76504
    frameStart := 76484 },
  { event := event76505
    frameStart := 76484 },
  { event := event76506
    frameStart := 76484 },
  { event := event76507
    frameStart := 76484 },
  { event := event76508
    frameStart := 76484 },
  { event := event76509
    frameStart := 76484 },
  { event := event76510
    frameStart := 76484 },
  { event := event76511
    frameStart := 76484 }
]

def eventLeaf4782 : Array AnnotatedEvent := #[
  { event := event76512
    frameStart := 76484 },
  { event := event76513
    frameStart := 76484 },
  { event := event76514
    frameStart := 76484 },
  { event := event76515
    frameStart := 76484 },
  { event := event76516
    frameStart := 76484 },
  { event := event76517
    frameStart := 76484 },
  { event := event76518
    frameStart := 76484 },
  { event := event76519
    frameStart := 76484 },
  { event := event76520
    frameStart := 76484 },
  { event := event76521
    frameStart := 76484 },
  { event := event76522
    frameStart := 76484 },
  { event := event76523
    frameStart := 76484 },
  { event := event76524
    frameStart := 76484 },
  { event := event76525
    frameStart := 76484 },
  { event := event76526
    frameStart := 76484 },
  { event := event76527
    frameStart := 76484 }
]

def eventLeaf4783 : Array AnnotatedEvent := #[
  { event := event76528
    frameStart := 76484 },
  { event := event76529
    frameStart := 76484 },
  { event := event76530
    frameStart := 76484 },
  { event := event76531
    frameStart := 76484 },
  { event := event76532
    frameStart := 76532 },
  { event := event76533
    frameStart := 76532 },
  { event := event76534
    frameStart := 76532 },
  { event := event76535
    frameStart := 76532 },
  { event := event76536
    frameStart := 76532 },
  { event := event76537
    frameStart := 76532 },
  { event := event76538
    frameStart := 76532 },
  { event := event76539
    frameStart := 76532 },
  { event := event76540
    frameStart := 76532 },
  { event := event76541
    frameStart := 76532 },
  { event := event76542
    frameStart := 76532 },
  { event := event76543
    frameStart := 76532 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events298
