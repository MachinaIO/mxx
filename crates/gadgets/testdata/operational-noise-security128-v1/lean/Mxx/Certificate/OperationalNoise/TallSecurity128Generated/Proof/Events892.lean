import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events892

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event228352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55343⟩⟩) (.identity (.predecessor 0 228351 .coefficient))

def exact228353RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], []⟩, (1)⟩]

theorem exact228353RawTermsValid :
    exact228353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228353 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55343⟩⟩) exact228353RawTerms (.finite 12) 228352 .exactZero (none)

def event228354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact228355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228355RawTermsValid :
    exact228355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact228355RawTerms .large 228354 .exactZero (none)

def event228356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55344⟩⟩) 0 ⟨6908⟩ 228355

def event228357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55344⟩⟩) 1 ⟨55343⟩ 228353

def event228358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55344⟩⟩) (.product (.predecessor 0 228356 .coefficient) (.predecessor 1 228357 .coefficient) (⟨false, false, none, none, none⟩))

def event228359 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55344⟩⟩, .operator (⟨228355, 0⟩, ⟨228353, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228360RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228360RawTermsValid :
    exact228360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55344⟩⟩) exact228360RawTerms .large 228358 .exactZero (none)

def event228361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 228337

def event228362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact228363RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact228363RawTermsValid :
    exact228363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228363 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact228363RawTerms .large 228362 .exactZero (none)

def event228364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55345⟩⟩) 0 ⟨7184⟩ 228363

def event228365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55345⟩⟩) 1 ⟨55344⟩ 228360

def event228366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55345⟩⟩) (.sum [.predecessor 0 228364 .coefficient, .predecessor 1 228365 .coefficient])

def exact228367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228367RawTermsValid :
    exact228367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228367 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55345⟩⟩) exact228367RawTerms .large 228366 .exactZero (none)

def event228368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55902⟩⟩) 0 ⟨55345⟩ 228367

def event228369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55902⟩⟩) 1 ⟨55901⟩ 228344

def event228370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55902⟩⟩) (.product (.predecessor 0 228368 .coefficient) (.predecessor 1 228369 .coefficient) (⟨false, false, none, none, none⟩))

def event228371 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55902⟩⟩, .operator (⟨228367, 0⟩, ⟨228344, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩)

def event228372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55902⟩⟩, .operator (⟨228367, 1⟩, ⟨228344, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩)

def event228373 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55902⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55901⟩⟩) ⟨55132⟩ 228341)

def event228374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55902⟩⟩, .relation 228373 0, ⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (-1)⟩)

def exact228375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (-1)⟩]

theorem exact228375RawTermsValid :
    exact228375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55902⟩⟩) exact228375RawTerms .large 228370 .exactZero (none)

def event228376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54122⟩⟩) 0 ⟨53861⟩ 228333

def event228377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54122⟩⟩) (.authority (.programFamilyFact))

def exact228378RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], []⟩, (1)⟩]

theorem exact228378RawTermsValid :
    exact228378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54122⟩⟩) exact228378RawTerms (.finite 59) 228377 .exactZero (none)

def event228379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54124⟩⟩) 0 ⟨6908⟩ 228355

def event228380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54124⟩⟩) 1 ⟨54122⟩ 228378

def event228381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54124⟩⟩) (.product (.predecessor 0 228379 .coefficient) (.predecessor 1 228380 .coefficient) (⟨false, true, none, none, some 1⟩))

def event228382 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54124⟩⟩, .operator (⟨228355, 0⟩, ⟨228378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228383RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228383RawTermsValid :
    exact228383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228383 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54124⟩⟩) exact228383RawTerms .large 228381 .exactZero (none)

def event228384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 228337

def event228385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact228386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact228386RawTermsValid :
    exact228386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact228386RawTerms .large 228385 .exactZero (none)

def event228387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54125⟩⟩) 0 ⟨7208⟩ 228386

def event228388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54125⟩⟩) 1 ⟨54124⟩ 228383

def event228389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54125⟩⟩) (.sum [.predecessor 0 228387 .coefficient, .predecessor 1 228388 .coefficient])

def exact228390RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228390RawTermsValid :
    exact228390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54125⟩⟩) exact228390RawTerms .large 228389 .exactZero (none)

def event228391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55906⟩⟩) 0 ⟨54125⟩ 228390

def event228392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55906⟩⟩) 1 ⟨55902⟩ 228375

def event228393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55906⟩⟩) (.sum [.predecessor 0 228391 .coefficient, .predecessor 1 228392 .coefficient])

def exact228394RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228394RawTermsValid :
    exact228394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55906⟩⟩) exact228394RawTerms .large 228393 .exactZero (none)

def event228395 : Event := .preFoldPolynomial 228394 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact228396RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event228396 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨55906⟩⟩) 228395 exact228396RawTerms .large 228393 .exactZero (none)

def event228397 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53861⟩⟩) ⟨⟨87⟩, ⟨68⟩, ⟨135⟩⟩ ⟨228239, 228397⟩

def event228398 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54719⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩) (1) 0 2 (.universal 228397 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩) (none) 228396)

def event228399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54719⟩⟩, .relation 228398 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩)

def event228400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54719⟩⟩, .relation 228398 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩)

def event228401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54719⟩⟩, .relation 228398 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩)

def event228402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54719⟩⟩, .relation 228398 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact228403RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228403RawTermsValid :
    exact228403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54719⟩⟩) exact228403RawTerms .large 228235 (.finite 202072841853861888) (some (228237))

def event228404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55904⟩⟩) 0 ⟨54719⟩ 228403

def event228405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55904⟩⟩) 1 ⟨55903⟩ 228225

def event228406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55904⟩⟩) (.sum [.predecessor 0 228404 .coefficient, .predecessor 1 228405 .coefficient])

def event228407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55904⟩⟩, .operator (⟨228403, 0⟩, ⟨228225, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩)

def event228408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55904⟩⟩, .operator (⟨228403, 2⟩, ⟨228225, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (-1)⟩)

def event228409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55904⟩⟩) (.sum [.result 228403 .summary, .result 228225 .summary])

def exact228410RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228410RawTermsValid :
    exact228410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228410 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55904⟩⟩) exact228410RawTerms .large 228406 (.finite 32189789464712143775715074244608) (some (228409))

def event228411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52150⟩⟩) 0 ⟨50881⟩ 10882

def event228412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52150⟩⟩) (.authority (.programFamilyFact))

def event228413 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52150⟩⟩) (.finite 3720)

def event228414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52152⟩⟩) 0 ⟨7177⟩ 15500

def event228415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52152⟩⟩) 1 ⟨52150⟩ 228413

def event228416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52152⟩⟩) (.authority (.operator))

def exact228417RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52152⟩⟩]⟩, (1)⟩]

theorem exact228417RawTermsValid :
    exact228417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228417 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52152⟩⟩) exact228417RawTerms .large 228416 .exactZero (none)

def event228418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52921⟩⟩) 0 ⟨52152⟩ 228417

def event228419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52921⟩⟩) (.authority (.operator))

def exact228420RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52921⟩⟩]⟩, (1)⟩]

theorem exact228420RawTermsValid :
    exact228420RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228420 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52921⟩⟩) exact228420RawTerms (.finite 8192) 228419 .exactZero (none)

def event228421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52002⟩⟩) 0 ⟨50520⟩ 10876

def event228422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52002⟩⟩) (.authority (.programFamilyFact))

def event228423 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52002⟩⟩) (.finite 3720)

def event228424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52003⟩⟩) 0 ⟨7177⟩ 15500

def event228425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52003⟩⟩) 1 ⟨52002⟩ 228423

def event228426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52003⟩⟩) (.authority (.operator))

def exact228427RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (1)⟩]

theorem exact228427RawTermsValid :
    exact228427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52003⟩⟩) exact228427RawTerms .large 228426 .exactZero (none)

def event228428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52508⟩⟩) 0 ⟨52003⟩ 228427

def event228429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52508⟩⟩) (.authority (.operator))

def exact228430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩]

theorem exact228430RawTermsValid :
    exact228430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52508⟩⟩) exact228430RawTerms (.finite 8192) 228429 .exactZero (none)

def event228431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24519⟩⟩) 0 ⟨24518⟩ 10865

def event228432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24519⟩⟩) 1 ⟨6937⟩ 222153

def event228433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24519⟩⟩) (.tensor (.predecessor 0 228431 .coefficient) (.predecessor 1 228432 .coefficient) true false)

def event228434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24519⟩⟩, .operator (⟨10865, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228435RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228435RawTermsValid :
    exact228435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24519⟩⟩) exact228435RawTerms .large 228433 .exactZero (none)

def event228436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8500⟩⟩) 0 ⟨5579⟩ 222023

def event228437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8500⟩⟩) 1 ⟨7308⟩ 23593

def event228438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8500⟩⟩) (.product (.predecessor 0 228436 .coefficient) (.predecessor 1 228437 .coefficient) (⟨false, false, none, none, none⟩))

def event228439 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8500⟩⟩, .operator (⟨222023, 0⟩, ⟨23593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact228440RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact228440RawTermsValid :
    exact228440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228440 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8500⟩⟩) exact228440RawTerms .large 228438 .exactZero (none)

def event228441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24520⟩⟩) 0 ⟨8500⟩ 228440

def event228442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24520⟩⟩) 1 ⟨24519⟩ 228435

def event228443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24520⟩⟩) (.sum [.predecessor 0 228441 .coefficient, .predecessor 1 228442 .coefficient])

def exact228444RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228444RawTermsValid :
    exact228444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24520⟩⟩) exact228444RawTerms .large 228443 .exactZero (none)

def event228445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24521⟩⟩) 0 ⟨24520⟩ 228444

def event228446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24521⟩⟩) 1 ⟨134⟩ 23585

def event228447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24521⟩⟩) (.sum [.predecessor 0 228445 .coefficient, .predecessor 1 228446 .coefficient])

def event228448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24521⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩) [⟨.result 23585 .coefficient, false, none⟩])

def event228449 : Event := .survivorFold (1) 228448

def exact228450RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228450RawTermsValid :
    exact228450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24521⟩⟩) exact228450RawTerms .large 228447 (.finite 26) (some (228448))

def event228451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50521⟩⟩) 0 ⟨24521⟩ 228450

def event228452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50521⟩⟩) 1 ⟨50518⟩ 10868

def event228453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50521⟩⟩) (.product (.predecessor 0 228451 .coefficient) (.predecessor 1 228452 .coefficient) (⟨false, true, none, none, some 1⟩))

def event228454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50521⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩) [⟨.result 10868 .coefficient, true, some 1⟩])

def event228455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50521⟩⟩) (.product (.result 228450 .summary) (.transfer 228454) (⟨false, false, none, none, none⟩))

def event228456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50521⟩⟩, .operator (⟨228450, 1⟩, ⟨10868, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event228457 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50521⟩⟩, .operator (⟨228450, 0⟩, ⟨10868, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact228458RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact228458RawTermsValid :
    exact228458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50521⟩⟩) exact228458RawTerms .large 228453 (.finite 8519680) (some (228455))

def event228459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50522⟩⟩) 0 ⟨50518⟩ 10868

def event228460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50522⟩⟩) 1 ⟨6937⟩ 222153

def event228461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50522⟩⟩) (.tensor (.predecessor 0 228459 .coefficient) (.predecessor 1 228460 .coefficient) true false)

def event228462 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50522⟩⟩, .operator (⟨10868, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228463RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228463RawTermsValid :
    exact228463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50522⟩⟩) exact228463RawTerms .large 228461 .exactZero (none)

def event228464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8480⟩⟩) 0 ⟨5579⟩ 222023

def event228465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8480⟩⟩) 1 ⟨7288⟩ 23634

def event228466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8480⟩⟩) (.product (.predecessor 0 228464 .coefficient) (.predecessor 1 228465 .coefficient) (⟨false, false, none, none, none⟩))

def event228467 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8480⟩⟩, .operator (⟨222023, 0⟩, ⟨23634, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩)

def exact228468RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact228468RawTermsValid :
    exact228468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8480⟩⟩) exact228468RawTerms .large 228466 .exactZero (none)

def event228469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50523⟩⟩) 0 ⟨8480⟩ 228468

def event228470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50523⟩⟩) 1 ⟨50522⟩ 228463

def event228471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50523⟩⟩) (.sum [.predecessor 0 228469 .coefficient, .predecessor 1 228470 .coefficient])

def exact228472RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228472RawTermsValid :
    exact228472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50523⟩⟩) exact228472RawTerms .large 228471 .exactZero (none)

def event228473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50524⟩⟩) 0 ⟨50523⟩ 228472

def event228474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50524⟩⟩) 1 ⟨114⟩ 23626

def event228475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50524⟩⟩) (.sum [.predecessor 0 228473 .coefficient, .predecessor 1 228474 .coefficient])

def event228476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50524⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩) [⟨.result 23626 .coefficient, false, none⟩])

def event228477 : Event := .survivorFold (1) 228476

def exact228478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228478RawTermsValid :
    exact228478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50524⟩⟩) exact228478RawTerms .large 228475 (.finite 26) (some (228476))

def event228479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50525⟩⟩) 0 ⟨50524⟩ 228478

def event228480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50525⟩⟩) 1 ⟨9581⟩ 23623

def event228481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50525⟩⟩) (.product (.predecessor 0 228479 .coefficient) (.predecessor 1 228480 .coefficient) (⟨false, false, none, none, none⟩))

def event228482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50525⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) [⟨.result 23619 .coefficient, false, none⟩])

def event228483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50525⟩⟩) (.product (.result 228478 .summary) (.transfer 228482) (⟨false, false, none, none, none⟩))

def event228484 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50525⟩⟩, .operator (⟨228478, 1⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (-1)⟩)

def event228485 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50525⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593)

def event228486 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50525⟩⟩, .relation 228485 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩)

def event228487 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50525⟩⟩, .operator (⟨228478, 0⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact228488RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩]

theorem exact228488RawTermsValid :
    exact228488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50525⟩⟩) exact228488RawTerms .large 228481 (.finite 279172874240) (some (228483))

def event228489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50526⟩⟩) 0 ⟨50525⟩ 228488

def event228490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50526⟩⟩) 1 ⟨50521⟩ 228458

def event228491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50526⟩⟩) (.sum [.predecessor 0 228489 .coefficient, .predecessor 1 228490 .coefficient])

def event228492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50526⟩⟩, .operator (⟨228488, 1⟩, ⟨228458, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def event228493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50526⟩⟩) (.sum [.result 228488 .summary, .result 228458 .summary])

def exact228494RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228494RawTermsValid :
    exact228494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228494 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50526⟩⟩) exact228494RawTerms .large 228491 (.finite 279181393920) (some (228493))

def event228495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52509⟩⟩) 0 ⟨50526⟩ 228494

def event228496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52509⟩⟩) 1 ⟨52508⟩ 228430

def event228497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52509⟩⟩) (.product (.predecessor 0 228495 .coefficient) (.predecessor 1 228496 .coefficient) (⟨false, false, none, none, none⟩))

def event228498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52509⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) [⟨.result 228430 .coefficient, false, none⟩])

def event228499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52509⟩⟩) (.product (.result 228494 .summary) (.transfer 228498) (⟨false, false, none, none, none⟩))

def event228500 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52509⟩⟩, .operator (⟨228494, 1⟩, ⟨228430, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (-1)⟩)

def event228501 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52509⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52508⟩⟩) ⟨52003⟩ 228427)

def event228502 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52509⟩⟩, .relation 228501 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (-1)⟩)

def event228503 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52509⟩⟩, .operator (⟨228494, 0⟩, ⟨228430, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩)

def exact228504RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩, (-1)⟩]

theorem exact228504RawTermsValid :
    exact228504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52509⟩⟩) exact228504RawTerms .large 228497 (.finite 2997687391345233100800) (some (228499))

def event228505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51439⟩⟩) 0 ⟨50520⟩ 10876

def event228506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51439⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact228507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩]

theorem exact228507RawTermsValid :
    exact228507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51439⟩⟩) exact228507RawTerms (.finite 5647228698) 228506 .exactZero (none)

def event228508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51441⟩⟩) 0 ⟨51439⟩ 228507

def event228509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51441⟩⟩) 1 ⟨2370⟩ 4

def event228510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51441⟩⟩) (.scale (.predecessor 0 228508 .coefficient) (.value (.predecessor 1 228509 .coefficient)))

def exact228511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩]

theorem exact228511RawTermsValid :
    exact228511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51441⟩⟩) exact228511RawTerms (.finite 5647228698) 228510 .exactZero (none)

def event228512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51442⟩⟩) 0 ⟨5581⟩ 222245

def event228513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51442⟩⟩) 1 ⟨51441⟩ 228511

def event228514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51442⟩⟩) (.product (.predecessor 0 228512 .coefficient) (.predecessor 1 228513 .coefficient) (⟨false, false, none, none, none⟩))

def event228515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51442⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) [⟨.result 228507 .coefficient, false, none⟩])

def event228516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51442⟩⟩) (.product (.result 222245 .summary) (.transfer 228515) (⟨false, false, none, none, none⟩))

def event228517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51442⟩⟩, .operator (⟨222245, 0⟩, ⟨228511, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩)

def event228518 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51440⟩⟩)

def event228519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228520 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228522 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228524 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228526 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228526

def event228528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228524

def event228529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228527 .coefficient) (.value (.predecessor 1 228528 .coefficient)))

def event228530 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228530

def event228532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228522

def event228533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228531 .coefficient, .predecessor 1 228532 .coefficient])

def event228534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228534

def event228536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228520

def event228537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228536 .coefficient))

def event228538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24518⟩⟩) 0 ⟨5577⟩ 228538

def event228540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24518⟩⟩) (.authority (.programFamilyFact))

def exact228541RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩], []⟩, (1)⟩]

theorem exact228541RawTermsValid :
    exact228541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24518⟩⟩) exact228541RawTerms (.finite 10) 228540 .exactZero (none)

def event228542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50518⟩⟩) 0 ⟨5577⟩ 228538

def event228543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50518⟩⟩) (.authority (.programFamilyFact))

def exact228544RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228544RawTermsValid :
    exact228544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50518⟩⟩) exact228544RawTerms (.finite 10) 228543 .exactZero (none)

def event228545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 0 ⟨50518⟩ 228544

def event228546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 1 ⟨24518⟩ 228541

def event228547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.product (.predecessor 0 228545 .coefficient) (.predecessor 1 228546 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩) [⟨.result 228544 .coefficient, true, some 1⟩, ⟨.result 228541 .coefficient, true, some 1⟩])

def event228549 : Event := .survivorFold (1) 228548

def exact228550RawTerms : List Term := []

theorem exact228550RawTermsValid :
    exact228550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50519⟩⟩) exact228550RawTerms (.finite 100) 228547 (.finite 100) (some (228548))

def event228551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50520⟩⟩) 0 ⟨50519⟩ 228550

def event228552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.identity (.predecessor 0 228551 .coefficient))

def event228553 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.finite 100)

def event228554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51439⟩⟩) 0 ⟨50520⟩ 228553

def event228555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51439⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact228556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩]

theorem exact228556RawTermsValid :
    exact228556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51439⟩⟩) exact228556RawTerms (.finite 5647228698) 228555 .exactZero (none)

def event228557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact228558RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact228558RawTermsValid :
    exact228558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228558 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact228558RawTerms .large 228557 .exactZero (none)

def event228559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51440⟩⟩) 0 ⟨35⟩ 228558

def event228560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51440⟩⟩) 1 ⟨51439⟩ 228556

def event228561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51440⟩⟩) (.product (.predecessor 0 228559 .coefficient) (.predecessor 1 228560 .coefficient) (⟨false, false, none, none, none⟩))

def event228562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51440⟩⟩, .operator (⟨228558, 0⟩, ⟨228556, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩)

def exact228563RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩]

theorem exact228563RawTermsValid :
    exact228563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51440⟩⟩) exact228563RawTerms .large 228561 .exactZero (none)

def event228564 : Event := .preFoldPolynomial 228563 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩] .exactZero none

def exact228565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩, (1)⟩]

def event228565 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51440⟩⟩) 228564 exact228565RawTerms .large 228561 .exactZero (none)

def event228566 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨52512⟩⟩)

def event228567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228568 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228570 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228572 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228574 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228574

def event228576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228572

def event228577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228575 .coefficient) (.value (.predecessor 1 228576 .coefficient)))

def event228578 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228578

def event228580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228570

def event228581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228579 .coefficient, .predecessor 1 228580 .coefficient])

def event228582 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228582

def event228584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228568

def event228585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228584 .coefficient))

def event228586 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24518⟩⟩) 0 ⟨5577⟩ 228586

def event228588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24518⟩⟩) (.authority (.programFamilyFact))

def exact228589RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩], []⟩, (1)⟩]

theorem exact228589RawTermsValid :
    exact228589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24518⟩⟩) exact228589RawTerms (.finite 10) 228588 .exactZero (none)

def event228590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50518⟩⟩) 0 ⟨5577⟩ 228586

def event228591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50518⟩⟩) (.authority (.programFamilyFact))

def exact228592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228592RawTermsValid :
    exact228592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50518⟩⟩) exact228592RawTerms (.finite 10) 228591 .exactZero (none)

def event228593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 0 ⟨50518⟩ 228592

def event228594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50519⟩⟩) 1 ⟨24518⟩ 228589

def event228595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50519⟩⟩) (.product (.predecessor 0 228593 .coefficient) (.predecessor 1 228594 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228596 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50519⟩⟩, .operator (⟨228592, 0⟩, ⟨228589, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩)

def exact228597RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩, (1)⟩]

theorem exact228597RawTermsValid :
    exact228597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50519⟩⟩) exact228597RawTerms (.finite 100) 228595 .exactZero (none)

def event228598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50520⟩⟩) 0 ⟨50519⟩ 228597

def event228599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.identity (.predecessor 0 228598 .coefficient))

def event228600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50520⟩⟩) (.finite 100)

def event228601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52002⟩⟩) 0 ⟨50520⟩ 228600

def event228602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52002⟩⟩) (.authority (.programFamilyFact))

def event228603 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52002⟩⟩) (.finite 3720)

def event228604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event228605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52003⟩⟩) 0 ⟨7177⟩ 228604

def event228606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52003⟩⟩) 1 ⟨52002⟩ 228603

def event228607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52003⟩⟩) (.authority (.operator))

def eventLeaf14272 : Array AnnotatedEvent := #[
  { event := event228352
    frameStart := 228293 },
  { event := event228353
    frameStart := 228293 },
  { event := event228354
    frameStart := 228293 },
  { event := event228355
    frameStart := 228293 },
  { event := event228356
    frameStart := 228293 },
  { event := event228357
    frameStart := 228293 },
  { event := event228358
    frameStart := 228293 },
  { event := event228359
    frameStart := 228293 },
  { event := event228360
    frameStart := 228293 },
  { event := event228361
    frameStart := 228293 },
  { event := event228362
    frameStart := 228293 },
  { event := event228363
    frameStart := 228293 },
  { event := event228364
    frameStart := 228293 },
  { event := event228365
    frameStart := 228293 },
  { event := event228366
    frameStart := 228293 },
  { event := event228367
    frameStart := 228293 }
]

def eventLeaf14273 : Array AnnotatedEvent := #[
  { event := event228368
    frameStart := 228293 },
  { event := event228369
    frameStart := 228293 },
  { event := event228370
    frameStart := 228293 },
  { event := event228371
    frameStart := 228293 },
  { event := event228372
    frameStart := 228293 },
  { event := event228373
    frameStart := 228293 },
  { event := event228374
    frameStart := 228293 },
  { event := event228375
    frameStart := 228293 },
  { event := event228376
    frameStart := 228293 },
  { event := event228377
    frameStart := 228293 },
  { event := event228378
    frameStart := 228293 },
  { event := event228379
    frameStart := 228293 },
  { event := event228380
    frameStart := 228293 },
  { event := event228381
    frameStart := 228293 },
  { event := event228382
    frameStart := 228293 },
  { event := event228383
    frameStart := 228293 }
]

def eventLeaf14274 : Array AnnotatedEvent := #[
  { event := event228384
    frameStart := 228293 },
  { event := event228385
    frameStart := 228293 },
  { event := event228386
    frameStart := 228293 },
  { event := event228387
    frameStart := 228293 },
  { event := event228388
    frameStart := 228293 },
  { event := event228389
    frameStart := 228293 },
  { event := event228390
    frameStart := 228293 },
  { event := event228391
    frameStart := 228293 },
  { event := event228392
    frameStart := 228293 },
  { event := event228393
    frameStart := 228293 },
  { event := event228394
    frameStart := 228293 },
  { event := event228395
    frameStart := 228293 },
  { event := event228396
    frameStart := 228293 },
  { event := event228397
    frameStart := 0 },
  { event := event228398
    frameStart := 0 },
  { event := event228399
    frameStart := 0 }
]

def eventLeaf14275 : Array AnnotatedEvent := #[
  { event := event228400
    frameStart := 0 },
  { event := event228401
    frameStart := 0 },
  { event := event228402
    frameStart := 0 },
  { event := event228403
    frameStart := 0 },
  { event := event228404
    frameStart := 0 },
  { event := event228405
    frameStart := 0 },
  { event := event228406
    frameStart := 0 },
  { event := event228407
    frameStart := 0 },
  { event := event228408
    frameStart := 0 },
  { event := event228409
    frameStart := 0 },
  { event := event228410
    frameStart := 0 },
  { event := event228411
    frameStart := 0 },
  { event := event228412
    frameStart := 0 },
  { event := event228413
    frameStart := 0 },
  { event := event228414
    frameStart := 0 },
  { event := event228415
    frameStart := 0 }
]

def eventLeaf14276 : Array AnnotatedEvent := #[
  { event := event228416
    frameStart := 0 },
  { event := event228417
    frameStart := 0 },
  { event := event228418
    frameStart := 0 },
  { event := event228419
    frameStart := 0 },
  { event := event228420
    frameStart := 0 },
  { event := event228421
    frameStart := 0 },
  { event := event228422
    frameStart := 0 },
  { event := event228423
    frameStart := 0 },
  { event := event228424
    frameStart := 0 },
  { event := event228425
    frameStart := 0 },
  { event := event228426
    frameStart := 0 },
  { event := event228427
    frameStart := 0 },
  { event := event228428
    frameStart := 0 },
  { event := event228429
    frameStart := 0 },
  { event := event228430
    frameStart := 0 },
  { event := event228431
    frameStart := 0 }
]

def eventLeaf14277 : Array AnnotatedEvent := #[
  { event := event228432
    frameStart := 0 },
  { event := event228433
    frameStart := 0 },
  { event := event228434
    frameStart := 0 },
  { event := event228435
    frameStart := 0 },
  { event := event228436
    frameStart := 0 },
  { event := event228437
    frameStart := 0 },
  { event := event228438
    frameStart := 0 },
  { event := event228439
    frameStart := 0 },
  { event := event228440
    frameStart := 0 },
  { event := event228441
    frameStart := 0 },
  { event := event228442
    frameStart := 0 },
  { event := event228443
    frameStart := 0 },
  { event := event228444
    frameStart := 0 },
  { event := event228445
    frameStart := 0 },
  { event := event228446
    frameStart := 0 },
  { event := event228447
    frameStart := 0 }
]

def eventLeaf14278 : Array AnnotatedEvent := #[
  { event := event228448
    frameStart := 0 },
  { event := event228449
    frameStart := 0 },
  { event := event228450
    frameStart := 0 },
  { event := event228451
    frameStart := 0 },
  { event := event228452
    frameStart := 0 },
  { event := event228453
    frameStart := 0 },
  { event := event228454
    frameStart := 0 },
  { event := event228455
    frameStart := 0 },
  { event := event228456
    frameStart := 0 },
  { event := event228457
    frameStart := 0 },
  { event := event228458
    frameStart := 0 },
  { event := event228459
    frameStart := 0 },
  { event := event228460
    frameStart := 0 },
  { event := event228461
    frameStart := 0 },
  { event := event228462
    frameStart := 0 },
  { event := event228463
    frameStart := 0 }
]

def eventLeaf14279 : Array AnnotatedEvent := #[
  { event := event228464
    frameStart := 0 },
  { event := event228465
    frameStart := 0 },
  { event := event228466
    frameStart := 0 },
  { event := event228467
    frameStart := 0 },
  { event := event228468
    frameStart := 0 },
  { event := event228469
    frameStart := 0 },
  { event := event228470
    frameStart := 0 },
  { event := event228471
    frameStart := 0 },
  { event := event228472
    frameStart := 0 },
  { event := event228473
    frameStart := 0 },
  { event := event228474
    frameStart := 0 },
  { event := event228475
    frameStart := 0 },
  { event := event228476
    frameStart := 0 },
  { event := event228477
    frameStart := 0 },
  { event := event228478
    frameStart := 0 },
  { event := event228479
    frameStart := 0 }
]

def eventLeaf14280 : Array AnnotatedEvent := #[
  { event := event228480
    frameStart := 0 },
  { event := event228481
    frameStart := 0 },
  { event := event228482
    frameStart := 0 },
  { event := event228483
    frameStart := 0 },
  { event := event228484
    frameStart := 0 },
  { event := event228485
    frameStart := 0 },
  { event := event228486
    frameStart := 0 },
  { event := event228487
    frameStart := 0 },
  { event := event228488
    frameStart := 0 },
  { event := event228489
    frameStart := 0 },
  { event := event228490
    frameStart := 0 },
  { event := event228491
    frameStart := 0 },
  { event := event228492
    frameStart := 0 },
  { event := event228493
    frameStart := 0 },
  { event := event228494
    frameStart := 0 },
  { event := event228495
    frameStart := 0 }
]

def eventLeaf14281 : Array AnnotatedEvent := #[
  { event := event228496
    frameStart := 0 },
  { event := event228497
    frameStart := 0 },
  { event := event228498
    frameStart := 0 },
  { event := event228499
    frameStart := 0 },
  { event := event228500
    frameStart := 0 },
  { event := event228501
    frameStart := 0 },
  { event := event228502
    frameStart := 0 },
  { event := event228503
    frameStart := 0 },
  { event := event228504
    frameStart := 0 },
  { event := event228505
    frameStart := 0 },
  { event := event228506
    frameStart := 0 },
  { event := event228507
    frameStart := 0 },
  { event := event228508
    frameStart := 0 },
  { event := event228509
    frameStart := 0 },
  { event := event228510
    frameStart := 0 },
  { event := event228511
    frameStart := 0 }
]

def eventLeaf14282 : Array AnnotatedEvent := #[
  { event := event228512
    frameStart := 0 },
  { event := event228513
    frameStart := 0 },
  { event := event228514
    frameStart := 0 },
  { event := event228515
    frameStart := 0 },
  { event := event228516
    frameStart := 0 },
  { event := event228517
    frameStart := 0 },
  { event := event228518
    frameStart := 228518 },
  { event := event228519
    frameStart := 228518 },
  { event := event228520
    frameStart := 228518 },
  { event := event228521
    frameStart := 228518 },
  { event := event228522
    frameStart := 228518 },
  { event := event228523
    frameStart := 228518 },
  { event := event228524
    frameStart := 228518 },
  { event := event228525
    frameStart := 228518 },
  { event := event228526
    frameStart := 228518 },
  { event := event228527
    frameStart := 228518 }
]

def eventLeaf14283 : Array AnnotatedEvent := #[
  { event := event228528
    frameStart := 228518 },
  { event := event228529
    frameStart := 228518 },
  { event := event228530
    frameStart := 228518 },
  { event := event228531
    frameStart := 228518 },
  { event := event228532
    frameStart := 228518 },
  { event := event228533
    frameStart := 228518 },
  { event := event228534
    frameStart := 228518 },
  { event := event228535
    frameStart := 228518 },
  { event := event228536
    frameStart := 228518 },
  { event := event228537
    frameStart := 228518 },
  { event := event228538
    frameStart := 228518 },
  { event := event228539
    frameStart := 228518 },
  { event := event228540
    frameStart := 228518 },
  { event := event228541
    frameStart := 228518 },
  { event := event228542
    frameStart := 228518 },
  { event := event228543
    frameStart := 228518 }
]

def eventLeaf14284 : Array AnnotatedEvent := #[
  { event := event228544
    frameStart := 228518 },
  { event := event228545
    frameStart := 228518 },
  { event := event228546
    frameStart := 228518 },
  { event := event228547
    frameStart := 228518 },
  { event := event228548
    frameStart := 228518 },
  { event := event228549
    frameStart := 228518 },
  { event := event228550
    frameStart := 228518 },
  { event := event228551
    frameStart := 228518 },
  { event := event228552
    frameStart := 228518 },
  { event := event228553
    frameStart := 228518 },
  { event := event228554
    frameStart := 228518 },
  { event := event228555
    frameStart := 228518 },
  { event := event228556
    frameStart := 228518 },
  { event := event228557
    frameStart := 228518 },
  { event := event228558
    frameStart := 228518 },
  { event := event228559
    frameStart := 228518 }
]

def eventLeaf14285 : Array AnnotatedEvent := #[
  { event := event228560
    frameStart := 228518 },
  { event := event228561
    frameStart := 228518 },
  { event := event228562
    frameStart := 228518 },
  { event := event228563
    frameStart := 228518 },
  { event := event228564
    frameStart := 228518 },
  { event := event228565
    frameStart := 228518 },
  { event := event228566
    frameStart := 228566 },
  { event := event228567
    frameStart := 228566 },
  { event := event228568
    frameStart := 228566 },
  { event := event228569
    frameStart := 228566 },
  { event := event228570
    frameStart := 228566 },
  { event := event228571
    frameStart := 228566 },
  { event := event228572
    frameStart := 228566 },
  { event := event228573
    frameStart := 228566 },
  { event := event228574
    frameStart := 228566 },
  { event := event228575
    frameStart := 228566 }
]

def eventLeaf14286 : Array AnnotatedEvent := #[
  { event := event228576
    frameStart := 228566 },
  { event := event228577
    frameStart := 228566 },
  { event := event228578
    frameStart := 228566 },
  { event := event228579
    frameStart := 228566 },
  { event := event228580
    frameStart := 228566 },
  { event := event228581
    frameStart := 228566 },
  { event := event228582
    frameStart := 228566 },
  { event := event228583
    frameStart := 228566 },
  { event := event228584
    frameStart := 228566 },
  { event := event228585
    frameStart := 228566 },
  { event := event228586
    frameStart := 228566 },
  { event := event228587
    frameStart := 228566 },
  { event := event228588
    frameStart := 228566 },
  { event := event228589
    frameStart := 228566 },
  { event := event228590
    frameStart := 228566 },
  { event := event228591
    frameStart := 228566 }
]

def eventLeaf14287 : Array AnnotatedEvent := #[
  { event := event228592
    frameStart := 228566 },
  { event := event228593
    frameStart := 228566 },
  { event := event228594
    frameStart := 228566 },
  { event := event228595
    frameStart := 228566 },
  { event := event228596
    frameStart := 228566 },
  { event := event228597
    frameStart := 228566 },
  { event := event228598
    frameStart := 228566 },
  { event := event228599
    frameStart := 228566 },
  { event := event228600
    frameStart := 228566 },
  { event := event228601
    frameStart := 228566 },
  { event := event228602
    frameStart := 228566 },
  { event := event228603
    frameStart := 228566 },
  { event := event228604
    frameStart := 228566 },
  { event := event228605
    frameStart := 228566 },
  { event := event228606
    frameStart := 228566 },
  { event := event228607
    frameStart := 228566 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events892
