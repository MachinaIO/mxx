import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events013

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event3328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 3327

def event3329 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact3330RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact3330RawTermsValid :
    exact3330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3330 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact3330RawTerms (.finite 18) 3329 .exactZero (none)

def event3331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15937⟩⟩) 0 ⟨15936⟩ 3330

def event3332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.identity (.predecessor 0 3331 .coefficient))

def event3333 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.finite 18)

def event3334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15983⟩⟩) 0 ⟨15937⟩ 3333

def event3335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15983⟩⟩) (.authority (.programFamilyFact))

def exact3336RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩]

theorem exact3336RawTermsValid :
    exact3336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15983⟩⟩) exact3336RawTerms (.finite 61) 3335 .exactZero (none)

def event3337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 3083

def event3338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact3339RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact3339RawTermsValid :
    exact3339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3339 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact3339RawTerms (.finite 16) 3338 .exactZero (none)

def event3340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 3083

def event3341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact3342RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact3342RawTermsValid :
    exact3342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3342 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact3342RawTerms (.finite 16) 3341 .exactZero (none)

def event3343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 3342

def event3344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 3339

def event3345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 3343 .coefficient) (.predecessor 1 3344 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3346 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13982⟩⟩, .operator (⟨3342, 0⟩, ⟨3339, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩)

def exact3347RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact3347RawTermsValid :
    exact3347RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3347 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact3347RawTerms (.finite 256) 3345 .exactZero (none)

def event3348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 3347

def event3349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 3348 .coefficient))

def event3350 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event3351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 3350

def event3352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact3353RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact3353RawTermsValid :
    exact3353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3353 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact3353RawTerms (.finite 16) 3352 .exactZero (none)

def event3354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15818⟩⟩) 0 ⟨15817⟩ 3353

def event3355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.identity (.predecessor 0 3354 .coefficient))

def event3356 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.finite 16)

def event3357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15864⟩⟩) 0 ⟨15818⟩ 3356

def event3358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15864⟩⟩) (.authority (.programFamilyFact))

def exact3359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩]

theorem exact3359RawTermsValid :
    exact3359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15864⟩⟩) exact3359RawTerms (.finite 60) 3358 .exactZero (none)

def event3360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 3083

def event3361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact3362RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact3362RawTermsValid :
    exact3362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3362 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact3362RawTerms (.finite 12) 3361 .exactZero (none)

def event3363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 3083

def event3364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact3365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact3365RawTermsValid :
    exact3365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact3365RawTerms (.finite 12) 3364 .exactZero (none)

def event3366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 3365

def event3367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 3362

def event3368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 3366 .coefficient) (.predecessor 1 3367 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3369 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13765⟩⟩, .operator (⟨3365, 0⟩, ⟨3362, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩)

def exact3370RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact3370RawTermsValid :
    exact3370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact3370RawTerms (.finite 144) 3368 .exactZero (none)

def event3371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 3370

def event3372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 3371 .coefficient))

def event3373 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event3374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 3373

def event3375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact3376RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact3376RawTermsValid :
    exact3376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact3376RawTerms (.finite 12) 3375 .exactZero (none)

def event3377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15699⟩⟩) 0 ⟨15698⟩ 3376

def event3378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.identity (.predecessor 0 3377 .coefficient))

def event3379 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.finite 12)

def event3380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15745⟩⟩) 0 ⟨15699⟩ 3379

def event3381 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15745⟩⟩) (.authority (.programFamilyFact))

def exact3382RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩]

theorem exact3382RawTermsValid :
    exact3382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3382 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15745⟩⟩) exact3382RawTerms (.finite 59) 3381 .exactZero (none)

def event3383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 3083

def event3384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact3385RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact3385RawTermsValid :
    exact3385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact3385RawTerms (.finite 10) 3384 .exactZero (none)

def event3386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 3083

def event3387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact3388RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact3388RawTermsValid :
    exact3388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3388 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact3388RawTerms (.finite 10) 3387 .exactZero (none)

def event3389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 3388

def event3390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 3385

def event3391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 3389 .coefficient) (.predecessor 1 3390 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3392 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13548⟩⟩, .operator (⟨3388, 0⟩, ⟨3385, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩)

def exact3393RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact3393RawTermsValid :
    exact3393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3393 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact3393RawTerms (.finite 100) 3391 .exactZero (none)

def event3394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 3393

def event3395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 3394 .coefficient))

def event3396 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event3397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15579⟩⟩) 0 ⟨13549⟩ 3396

def event3398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15579⟩⟩) (.authority (.programFamilyFact))

def exact3399RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact3399RawTermsValid :
    exact3399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3399 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15579⟩⟩) exact3399RawTerms (.finite 10) 3398 .exactZero (none)

def event3400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15580⟩⟩) 0 ⟨15579⟩ 3399

def event3401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.identity (.predecessor 0 3400 .coefficient))

def event3402 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.finite 10)

def event3403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15626⟩⟩) 0 ⟨15580⟩ 3402

def event3404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15626⟩⟩) (.authority (.programFamilyFact))

def exact3405RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩]

theorem exact3405RawTermsValid :
    exact3405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15626⟩⟩) exact3405RawTerms (.finite 58) 3404 .exactZero (none)

def event3406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 3083

def event3407 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact3408RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact3408RawTermsValid :
    exact3408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3408 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact3408RawTerms (.finite 6) 3407 .exactZero (none)

def event3409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 3083

def event3410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact3411RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact3411RawTermsValid :
    exact3411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact3411RawTerms (.finite 6) 3410 .exactZero (none)

def event3412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 3411

def event3413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 3408

def event3414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 3412 .coefficient) (.predecessor 1 3413 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3415 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12155⟩⟩, .operator (⟨3411, 0⟩, ⟨3408, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩)

def exact3416RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact3416RawTermsValid :
    exact3416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact3416RawTerms (.finite 36) 3414 .exactZero (none)

def event3417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 3416

def event3418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 3417 .coefficient))

def event3419 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event3420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 3419

def event3421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact3422RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact3422RawTermsValid :
    exact3422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact3422RawTerms (.finite 6) 3421 .exactZero (none)

def event3423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15419⟩⟩) 0 ⟨15418⟩ 3422

def event3424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.identity (.predecessor 0 3423 .coefficient))

def event3425 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.finite 6)

def event3426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17318⟩⟩) 0 ⟨15419⟩ 3425

def event3427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17318⟩⟩) (.authority (.programFamilyFact))

def exact3428RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3428RawTermsValid :
    exact3428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3428 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17318⟩⟩) exact3428RawTerms (.finite 55) 3427 .exactZero (none)

def event3429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10969⟩⟩) 0 ⟨5530⟩ 3083

def event3430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10969⟩⟩) (.authority (.programFamilyFact))

def exact3431RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact3431RawTermsValid :
    exact3431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10969⟩⟩) exact3431RawTerms (.finite 4) 3430 .exactZero (none)

def event3432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10837⟩⟩) 0 ⟨5530⟩ 3083

def event3433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10837⟩⟩) (.authority (.programFamilyFact))

def exact3434RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩, (1)⟩]

theorem exact3434RawTermsValid :
    exact3434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3434 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10837⟩⟩) exact3434RawTerms (.finite 4) 3433 .exactZero (none)

def event3435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 0 ⟨10837⟩ 3434

def event3436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 1 ⟨10969⟩ 3431

def event3437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.product (.predecessor 0 3435 .coefficient) (.predecessor 1 3436 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3438 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10970⟩⟩, .operator (⟨3434, 0⟩, ⟨3431, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩)

def exact3439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact3439RawTermsValid :
    exact3439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10970⟩⟩) exact3439RawTerms (.finite 16) 3437 .exactZero (none)

def event3440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 3439

def event3441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 3440 .coefficient))

def event3442 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event3443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 3442

def event3444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact3445RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact3445RawTermsValid :
    exact3445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact3445RawTerms (.finite 4) 3444 .exactZero (none)

def event3446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15111⟩⟩) 0 ⟨15110⟩ 3445

def event3447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.identity (.predecessor 0 3446 .coefficient))

def event3448 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.finite 4)

def event3449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15362⟩⟩) 0 ⟨15111⟩ 3448

def event3450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15362⟩⟩) (.authority (.programFamilyFact))

def exact3451RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩]

theorem exact3451RawTermsValid :
    exact3451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3451 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15362⟩⟩) exact3451RawTerms (.finite 51) 3450 .exactZero (none)

def event3452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10668⟩⟩) 0 ⟨5530⟩ 3083

def event3453 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10668⟩⟩) (.authority (.programFamilyFact))

def exact3454RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact3454RawTermsValid :
    exact3454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3454 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10668⟩⟩) exact3454RawTerms (.finite 3) 3453 .exactZero (none)

def event3455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9500⟩⟩) 0 ⟨5530⟩ 3083

def event3456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9500⟩⟩) (.authority (.programFamilyFact))

def exact3457RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩, (1)⟩]

theorem exact3457RawTermsValid :
    exact3457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9500⟩⟩) exact3457RawTerms (.finite 3) 3456 .exactZero (none)

def event3458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 0 ⟨9500⟩ 3457

def event3459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 1 ⟨10668⟩ 3454

def event3460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.product (.predecessor 0 3458 .coefficient) (.predecessor 1 3459 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3461 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10669⟩⟩, .operator (⟨3457, 0⟩, ⟨3454, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩)

def exact3462RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact3462RawTermsValid :
    exact3462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10669⟩⟩) exact3462RawTerms (.finite 9) 3460 .exactZero (none)

def event3463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10670⟩⟩) 0 ⟨10669⟩ 3462

def event3464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.identity (.predecessor 0 3463 .coefficient))

def event3465 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.finite 9)

def event3466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14949⟩⟩) 0 ⟨10670⟩ 3465

def event3467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14949⟩⟩) (.authority (.programFamilyFact))

def exact3468RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], []⟩, (1)⟩]

theorem exact3468RawTermsValid :
    exact3468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3468 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14949⟩⟩) exact3468RawTerms (.finite 3) 3467 .exactZero (none)

def event3469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14950⟩⟩) 0 ⟨14949⟩ 3468

def event3470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.identity (.predecessor 0 3469 .coefficient))

def event3471 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.finite 3)

def event3472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15306⟩⟩) 0 ⟨14950⟩ 3471

def event3473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact3474RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact3474RawTermsValid :
    exact3474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15306⟩⟩) exact3474RawTerms (.finite 48) 3473 .exactZero (none)

def event3475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 3083

def event3476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact3477RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact3477RawTermsValid :
    exact3477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3477 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact3477RawTerms (.finite 2) 3476 .exactZero (none)

def event3478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 3083

def event3479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact3480RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact3480RawTermsValid :
    exact3480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3480 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact3480RawTerms (.finite 2) 3479 .exactZero (none)

def event3481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 3480

def event3482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 3477

def event3483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 3481 .coefficient) (.predecessor 1 3482 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3484 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10473⟩⟩, .operator (⟨3480, 0⟩, ⟨3477, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩)

def exact3485RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact3485RawTermsValid :
    exact3485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact3485RawTerms (.finite 4) 3483 .exactZero (none)

def event3486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 3485

def event3487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 3486 .coefficient))

def event3488 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event3489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 3488

def event3490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact3491RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact3491RawTermsValid :
    exact3491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact3491RawTerms (.finite 2) 3490 .exactZero (none)

def event3492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14789⟩⟩) 0 ⟨14788⟩ 3491

def event3493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.identity (.predecessor 0 3492 .coefficient))

def event3494 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.finite 2)

def event3495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15262⟩⟩) 0 ⟨14789⟩ 3494

def event3496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15262⟩⟩) (.authority (.programFamilyFact))

def exact3497RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩]

theorem exact3497RawTermsValid :
    exact3497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15262⟩⟩) exact3497RawTerms (.finite 43) 3496 .exactZero (none)

def event3498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15307⟩⟩) 0 ⟨15262⟩ 3497

def event3499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 3474

def event3500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15307⟩⟩) (.sum [.predecessor 0 3498 .coefficient, .predecessor 1 3499 .coefficient])

def exact3501RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact3501RawTermsValid :
    exact3501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3501 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15307⟩⟩) exact3501RawTerms (.finite 91) 3500 .exactZero (none)

def event3502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15363⟩⟩) 0 ⟨15307⟩ 3501

def event3503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15363⟩⟩) 1 ⟨15362⟩ 3451

def event3504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15363⟩⟩) (.sum [.predecessor 0 3502 .coefficient, .predecessor 1 3503 .coefficient])

def exact3505RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩]

theorem exact3505RawTermsValid :
    exact3505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15363⟩⟩) exact3505RawTerms (.finite 142) 3504 .exactZero (none)

def event3506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17319⟩⟩) 0 ⟨15363⟩ 3505

def event3507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17319⟩⟩) 1 ⟨17318⟩ 3428

def event3508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17319⟩⟩) (.sum [.predecessor 0 3506 .coefficient, .predecessor 1 3507 .coefficient])

def exact3509RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3509RawTermsValid :
    exact3509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17319⟩⟩) exact3509RawTerms (.finite 197) 3508 .exactZero (none)

def event3510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17320⟩⟩) 0 ⟨17319⟩ 3509

def event3511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17320⟩⟩) 1 ⟨15626⟩ 3405

def event3512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17320⟩⟩) (.sum [.predecessor 0 3510 .coefficient, .predecessor 1 3511 .coefficient])

def exact3513RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3513RawTermsValid :
    exact3513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3513 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17320⟩⟩) exact3513RawTerms (.finite 255) 3512 .exactZero (none)

def event3514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17321⟩⟩) 0 ⟨17320⟩ 3513

def event3515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17321⟩⟩) 1 ⟨15745⟩ 3382

def event3516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17321⟩⟩) (.sum [.predecessor 0 3514 .coefficient, .predecessor 1 3515 .coefficient])

def exact3517RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3517RawTermsValid :
    exact3517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17321⟩⟩) exact3517RawTerms (.finite 314) 3516 .exactZero (none)

def event3518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17322⟩⟩) 0 ⟨17321⟩ 3517

def event3519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17322⟩⟩) 1 ⟨15864⟩ 3359

def event3520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17322⟩⟩) (.sum [.predecessor 0 3518 .coefficient, .predecessor 1 3519 .coefficient])

def exact3521RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3521RawTermsValid :
    exact3521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17322⟩⟩) exact3521RawTerms (.finite 374) 3520 .exactZero (none)

def event3522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17323⟩⟩) 0 ⟨17322⟩ 3521

def event3523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17323⟩⟩) 1 ⟨15983⟩ 3336

def event3524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17323⟩⟩) (.sum [.predecessor 0 3522 .coefficient, .predecessor 1 3523 .coefficient])

def exact3525RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3525RawTermsValid :
    exact3525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17323⟩⟩) exact3525RawTerms (.finite 435) 3524 .exactZero (none)

def event3526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17324⟩⟩) 0 ⟨17323⟩ 3525

def event3527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17324⟩⟩) 1 ⟨16102⟩ 3313

def event3528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17324⟩⟩) (.sum [.predecessor 0 3526 .coefficient, .predecessor 1 3527 .coefficient])

def exact3529RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩]

theorem exact3529RawTermsValid :
    exact3529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3529 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17324⟩⟩) exact3529RawTerms (.finite 496) 3528 .exactZero (none)

def event3530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18328⟩⟩) 0 ⟨17324⟩ 3529

def event3531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18328⟩⟩) 1 ⟨18327⟩ 3290

def event3532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18328⟩⟩) (.sum [.predecessor 0 3530 .coefficient, .predecessor 1 3531 .coefficient])

def exact3533RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3533RawTermsValid :
    exact3533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18328⟩⟩) exact3533RawTerms (.finite 558) 3532 .exactZero (none)

def event3534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18329⟩⟩) 0 ⟨18328⟩ 3533

def event3535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18329⟩⟩) 1 ⟨16305⟩ 3267

def event3536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18329⟩⟩) (.sum [.predecessor 0 3534 .coefficient, .predecessor 1 3535 .coefficient])

def exact3537RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3537RawTermsValid :
    exact3537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18329⟩⟩) exact3537RawTerms (.finite 620) 3536 .exactZero (none)

def event3538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18330⟩⟩) 0 ⟨18329⟩ 3537

def event3539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18330⟩⟩) 1 ⟨17117⟩ 3244

def event3540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18330⟩⟩) (.sum [.predecessor 0 3538 .coefficient, .predecessor 1 3539 .coefficient])

def exact3541RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3541RawTermsValid :
    exact3541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3541 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18330⟩⟩) exact3541RawTerms (.finite 682) 3540 .exactZero (none)

def event3542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18331⟩⟩) 0 ⟨18330⟩ 3541

def event3543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18331⟩⟩) 1 ⟨17901⟩ 3221

def event3544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18331⟩⟩) (.sum [.predecessor 0 3542 .coefficient, .predecessor 1 3543 .coefficient])

def exact3545RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3545RawTermsValid :
    exact3545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18331⟩⟩) exact3545RawTerms (.finite 744) 3544 .exactZero (none)

def event3546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18332⟩⟩) 0 ⟨18331⟩ 3545

def event3547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18332⟩⟩) 1 ⟨18202⟩ 3198

def event3548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18332⟩⟩) (.sum [.predecessor 0 3546 .coefficient, .predecessor 1 3547 .coefficient])

def exact3549RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18202⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3549RawTermsValid :
    exact3549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18332⟩⟩) exact3549RawTerms (.finite 807) 3548 .exactZero (none)

def event3550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18333⟩⟩) 0 ⟨18332⟩ 3549

def event3551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18333⟩⟩) 1 ⟨16676⟩ 3175

def event3552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18333⟩⟩) (.sum [.predecessor 0 3550 .coefficient, .predecessor 1 3551 .coefficient])

def exact3553RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16676⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18202⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3553RawTermsValid :
    exact3553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3553 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18333⟩⟩) exact3553RawTerms (.finite 870) 3552 .exactZero (none)

def event3554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18334⟩⟩) 0 ⟨18333⟩ 3553

def event3555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18334⟩⟩) 1 ⟨16795⟩ 3152

def event3556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18334⟩⟩) (.sum [.predecessor 0 3554 .coefficient, .predecessor 1 3555 .coefficient])

def exact3557RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16676⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18202⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3557RawTermsValid :
    exact3557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18334⟩⟩) exact3557RawTerms (.finite 933) 3556 .exactZero (none)

def event3558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18335⟩⟩) 0 ⟨18334⟩ 3557

def event3559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18335⟩⟩) 1 ⟨17082⟩ 3129

def event3560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18335⟩⟩) (.sum [.predecessor 0 3558 .coefficient, .predecessor 1 3559 .coefficient])

def exact3561RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16676⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17082⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18202⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3561RawTermsValid :
    exact3561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18335⟩⟩) exact3561RawTerms (.finite 996) 3560 .exactZero (none)

def event3562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18336⟩⟩) 0 ⟨18335⟩ 3561

def event3563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18336⟩⟩) 1 ⟨18167⟩ 3106

def event3564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18336⟩⟩) (.sum [.predecessor 0 3562 .coefficient, .predecessor 1 3563 .coefficient])

def exact3565RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16676⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17082⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18167⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18202⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact3565RawTermsValid :
    exact3565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3565 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18336⟩⟩) exact3565RawTerms (.finite 1059) 3564 .exactZero (none)

def event3566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18337⟩⟩) 0 ⟨18336⟩ 3565

def event3567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18337⟩⟩) (.identity (.predecessor 0 3566 .coefficient))

def event3568 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨18337⟩⟩) (.finite 1059)

def event3569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18491⟩⟩) 0 ⟨18337⟩ 3568

def event3570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18491⟩⟩) (.authority (.programFamilyFact))

def exact3571RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (1)⟩]

theorem exact3571RawTermsValid :
    exact3571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3571 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18491⟩⟩) exact3571RawTerms (.finite 18) 3570 .exactZero (none)

def event3572 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18492⟩⟩) 0 ⟨18491⟩ 3571

def event3573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18492⟩⟩) 1 ⟨6410⟩ 36

def event3574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18492⟩⟩) (.product (.predecessor 0 3572 .coefficient) (.predecessor 1 3573 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3575 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18492⟩⟩, .operator (⟨3571, 0⟩, ⟨36, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (1)⟩)

def exact3576RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], []⟩, (1)⟩]

theorem exact3576RawTermsValid :
    exact3576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18492⟩⟩) exact3576RawTerms (.finite 4222381728938650955397720) 3574 .exactZero (none)

def event3577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18120⟩⟩) 0 ⟨17008⟩ 3103

def event3578 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18120⟩⟩) (.authority (.programFamilyFact))

def exact3579RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩]

theorem exact3579RawTermsValid :
    exact3579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3579 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18120⟩⟩) exact3579RawTerms (.finite 60) 3578 .exactZero (none)

def event3580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18121⟩⟩) 0 ⟨18120⟩ 3579

def event3581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18121⟩⟩) 1 ⟨6435⟩ 543

def event3582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18121⟩⟩) (.product (.predecessor 0 3580 .coefficient) (.predecessor 1 3581 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3583 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18121⟩⟩, .operator (⟨3579, 0⟩, ⟨543, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], []⟩, (1)⟩)

def eventLeaf208 : Array AnnotatedEvent := #[
  { event := event3328
    frameStart := 0 },
  { event := event3329
    frameStart := 0 },
  { event := event3330
    frameStart := 0 },
  { event := event3331
    frameStart := 0 },
  { event := event3332
    frameStart := 0 },
  { event := event3333
    frameStart := 0 },
  { event := event3334
    frameStart := 0 },
  { event := event3335
    frameStart := 0 },
  { event := event3336
    frameStart := 0 },
  { event := event3337
    frameStart := 0 },
  { event := event3338
    frameStart := 0 },
  { event := event3339
    frameStart := 0 },
  { event := event3340
    frameStart := 0 },
  { event := event3341
    frameStart := 0 },
  { event := event3342
    frameStart := 0 },
  { event := event3343
    frameStart := 0 }
]

def eventLeaf209 : Array AnnotatedEvent := #[
  { event := event3344
    frameStart := 0 },
  { event := event3345
    frameStart := 0 },
  { event := event3346
    frameStart := 0 },
  { event := event3347
    frameStart := 0 },
  { event := event3348
    frameStart := 0 },
  { event := event3349
    frameStart := 0 },
  { event := event3350
    frameStart := 0 },
  { event := event3351
    frameStart := 0 },
  { event := event3352
    frameStart := 0 },
  { event := event3353
    frameStart := 0 },
  { event := event3354
    frameStart := 0 },
  { event := event3355
    frameStart := 0 },
  { event := event3356
    frameStart := 0 },
  { event := event3357
    frameStart := 0 },
  { event := event3358
    frameStart := 0 },
  { event := event3359
    frameStart := 0 }
]

def eventLeaf210 : Array AnnotatedEvent := #[
  { event := event3360
    frameStart := 0 },
  { event := event3361
    frameStart := 0 },
  { event := event3362
    frameStart := 0 },
  { event := event3363
    frameStart := 0 },
  { event := event3364
    frameStart := 0 },
  { event := event3365
    frameStart := 0 },
  { event := event3366
    frameStart := 0 },
  { event := event3367
    frameStart := 0 },
  { event := event3368
    frameStart := 0 },
  { event := event3369
    frameStart := 0 },
  { event := event3370
    frameStart := 0 },
  { event := event3371
    frameStart := 0 },
  { event := event3372
    frameStart := 0 },
  { event := event3373
    frameStart := 0 },
  { event := event3374
    frameStart := 0 },
  { event := event3375
    frameStart := 0 }
]

def eventLeaf211 : Array AnnotatedEvent := #[
  { event := event3376
    frameStart := 0 },
  { event := event3377
    frameStart := 0 },
  { event := event3378
    frameStart := 0 },
  { event := event3379
    frameStart := 0 },
  { event := event3380
    frameStart := 0 },
  { event := event3381
    frameStart := 0 },
  { event := event3382
    frameStart := 0 },
  { event := event3383
    frameStart := 0 },
  { event := event3384
    frameStart := 0 },
  { event := event3385
    frameStart := 0 },
  { event := event3386
    frameStart := 0 },
  { event := event3387
    frameStart := 0 },
  { event := event3388
    frameStart := 0 },
  { event := event3389
    frameStart := 0 },
  { event := event3390
    frameStart := 0 },
  { event := event3391
    frameStart := 0 }
]

def eventLeaf212 : Array AnnotatedEvent := #[
  { event := event3392
    frameStart := 0 },
  { event := event3393
    frameStart := 0 },
  { event := event3394
    frameStart := 0 },
  { event := event3395
    frameStart := 0 },
  { event := event3396
    frameStart := 0 },
  { event := event3397
    frameStart := 0 },
  { event := event3398
    frameStart := 0 },
  { event := event3399
    frameStart := 0 },
  { event := event3400
    frameStart := 0 },
  { event := event3401
    frameStart := 0 },
  { event := event3402
    frameStart := 0 },
  { event := event3403
    frameStart := 0 },
  { event := event3404
    frameStart := 0 },
  { event := event3405
    frameStart := 0 },
  { event := event3406
    frameStart := 0 },
  { event := event3407
    frameStart := 0 }
]

def eventLeaf213 : Array AnnotatedEvent := #[
  { event := event3408
    frameStart := 0 },
  { event := event3409
    frameStart := 0 },
  { event := event3410
    frameStart := 0 },
  { event := event3411
    frameStart := 0 },
  { event := event3412
    frameStart := 0 },
  { event := event3413
    frameStart := 0 },
  { event := event3414
    frameStart := 0 },
  { event := event3415
    frameStart := 0 },
  { event := event3416
    frameStart := 0 },
  { event := event3417
    frameStart := 0 },
  { event := event3418
    frameStart := 0 },
  { event := event3419
    frameStart := 0 },
  { event := event3420
    frameStart := 0 },
  { event := event3421
    frameStart := 0 },
  { event := event3422
    frameStart := 0 },
  { event := event3423
    frameStart := 0 }
]

def eventLeaf214 : Array AnnotatedEvent := #[
  { event := event3424
    frameStart := 0 },
  { event := event3425
    frameStart := 0 },
  { event := event3426
    frameStart := 0 },
  { event := event3427
    frameStart := 0 },
  { event := event3428
    frameStart := 0 },
  { event := event3429
    frameStart := 0 },
  { event := event3430
    frameStart := 0 },
  { event := event3431
    frameStart := 0 },
  { event := event3432
    frameStart := 0 },
  { event := event3433
    frameStart := 0 },
  { event := event3434
    frameStart := 0 },
  { event := event3435
    frameStart := 0 },
  { event := event3436
    frameStart := 0 },
  { event := event3437
    frameStart := 0 },
  { event := event3438
    frameStart := 0 },
  { event := event3439
    frameStart := 0 }
]

def eventLeaf215 : Array AnnotatedEvent := #[
  { event := event3440
    frameStart := 0 },
  { event := event3441
    frameStart := 0 },
  { event := event3442
    frameStart := 0 },
  { event := event3443
    frameStart := 0 },
  { event := event3444
    frameStart := 0 },
  { event := event3445
    frameStart := 0 },
  { event := event3446
    frameStart := 0 },
  { event := event3447
    frameStart := 0 },
  { event := event3448
    frameStart := 0 },
  { event := event3449
    frameStart := 0 },
  { event := event3450
    frameStart := 0 },
  { event := event3451
    frameStart := 0 },
  { event := event3452
    frameStart := 0 },
  { event := event3453
    frameStart := 0 },
  { event := event3454
    frameStart := 0 },
  { event := event3455
    frameStart := 0 }
]

def eventLeaf216 : Array AnnotatedEvent := #[
  { event := event3456
    frameStart := 0 },
  { event := event3457
    frameStart := 0 },
  { event := event3458
    frameStart := 0 },
  { event := event3459
    frameStart := 0 },
  { event := event3460
    frameStart := 0 },
  { event := event3461
    frameStart := 0 },
  { event := event3462
    frameStart := 0 },
  { event := event3463
    frameStart := 0 },
  { event := event3464
    frameStart := 0 },
  { event := event3465
    frameStart := 0 },
  { event := event3466
    frameStart := 0 },
  { event := event3467
    frameStart := 0 },
  { event := event3468
    frameStart := 0 },
  { event := event3469
    frameStart := 0 },
  { event := event3470
    frameStart := 0 },
  { event := event3471
    frameStart := 0 }
]

def eventLeaf217 : Array AnnotatedEvent := #[
  { event := event3472
    frameStart := 0 },
  { event := event3473
    frameStart := 0 },
  { event := event3474
    frameStart := 0 },
  { event := event3475
    frameStart := 0 },
  { event := event3476
    frameStart := 0 },
  { event := event3477
    frameStart := 0 },
  { event := event3478
    frameStart := 0 },
  { event := event3479
    frameStart := 0 },
  { event := event3480
    frameStart := 0 },
  { event := event3481
    frameStart := 0 },
  { event := event3482
    frameStart := 0 },
  { event := event3483
    frameStart := 0 },
  { event := event3484
    frameStart := 0 },
  { event := event3485
    frameStart := 0 },
  { event := event3486
    frameStart := 0 },
  { event := event3487
    frameStart := 0 }
]

def eventLeaf218 : Array AnnotatedEvent := #[
  { event := event3488
    frameStart := 0 },
  { event := event3489
    frameStart := 0 },
  { event := event3490
    frameStart := 0 },
  { event := event3491
    frameStart := 0 },
  { event := event3492
    frameStart := 0 },
  { event := event3493
    frameStart := 0 },
  { event := event3494
    frameStart := 0 },
  { event := event3495
    frameStart := 0 },
  { event := event3496
    frameStart := 0 },
  { event := event3497
    frameStart := 0 },
  { event := event3498
    frameStart := 0 },
  { event := event3499
    frameStart := 0 },
  { event := event3500
    frameStart := 0 },
  { event := event3501
    frameStart := 0 },
  { event := event3502
    frameStart := 0 },
  { event := event3503
    frameStart := 0 }
]

def eventLeaf219 : Array AnnotatedEvent := #[
  { event := event3504
    frameStart := 0 },
  { event := event3505
    frameStart := 0 },
  { event := event3506
    frameStart := 0 },
  { event := event3507
    frameStart := 0 },
  { event := event3508
    frameStart := 0 },
  { event := event3509
    frameStart := 0 },
  { event := event3510
    frameStart := 0 },
  { event := event3511
    frameStart := 0 },
  { event := event3512
    frameStart := 0 },
  { event := event3513
    frameStart := 0 },
  { event := event3514
    frameStart := 0 },
  { event := event3515
    frameStart := 0 },
  { event := event3516
    frameStart := 0 },
  { event := event3517
    frameStart := 0 },
  { event := event3518
    frameStart := 0 },
  { event := event3519
    frameStart := 0 }
]

def eventLeaf220 : Array AnnotatedEvent := #[
  { event := event3520
    frameStart := 0 },
  { event := event3521
    frameStart := 0 },
  { event := event3522
    frameStart := 0 },
  { event := event3523
    frameStart := 0 },
  { event := event3524
    frameStart := 0 },
  { event := event3525
    frameStart := 0 },
  { event := event3526
    frameStart := 0 },
  { event := event3527
    frameStart := 0 },
  { event := event3528
    frameStart := 0 },
  { event := event3529
    frameStart := 0 },
  { event := event3530
    frameStart := 0 },
  { event := event3531
    frameStart := 0 },
  { event := event3532
    frameStart := 0 },
  { event := event3533
    frameStart := 0 },
  { event := event3534
    frameStart := 0 },
  { event := event3535
    frameStart := 0 }
]

def eventLeaf221 : Array AnnotatedEvent := #[
  { event := event3536
    frameStart := 0 },
  { event := event3537
    frameStart := 0 },
  { event := event3538
    frameStart := 0 },
  { event := event3539
    frameStart := 0 },
  { event := event3540
    frameStart := 0 },
  { event := event3541
    frameStart := 0 },
  { event := event3542
    frameStart := 0 },
  { event := event3543
    frameStart := 0 },
  { event := event3544
    frameStart := 0 },
  { event := event3545
    frameStart := 0 },
  { event := event3546
    frameStart := 0 },
  { event := event3547
    frameStart := 0 },
  { event := event3548
    frameStart := 0 },
  { event := event3549
    frameStart := 0 },
  { event := event3550
    frameStart := 0 },
  { event := event3551
    frameStart := 0 }
]

def eventLeaf222 : Array AnnotatedEvent := #[
  { event := event3552
    frameStart := 0 },
  { event := event3553
    frameStart := 0 },
  { event := event3554
    frameStart := 0 },
  { event := event3555
    frameStart := 0 },
  { event := event3556
    frameStart := 0 },
  { event := event3557
    frameStart := 0 },
  { event := event3558
    frameStart := 0 },
  { event := event3559
    frameStart := 0 },
  { event := event3560
    frameStart := 0 },
  { event := event3561
    frameStart := 0 },
  { event := event3562
    frameStart := 0 },
  { event := event3563
    frameStart := 0 },
  { event := event3564
    frameStart := 0 },
  { event := event3565
    frameStart := 0 },
  { event := event3566
    frameStart := 0 },
  { event := event3567
    frameStart := 0 }
]

def eventLeaf223 : Array AnnotatedEvent := #[
  { event := event3568
    frameStart := 0 },
  { event := event3569
    frameStart := 0 },
  { event := event3570
    frameStart := 0 },
  { event := event3571
    frameStart := 0 },
  { event := event3572
    frameStart := 0 },
  { event := event3573
    frameStart := 0 },
  { event := event3574
    frameStart := 0 },
  { event := event3575
    frameStart := 0 },
  { event := event3576
    frameStart := 0 },
  { event := event3577
    frameStart := 0 },
  { event := event3578
    frameStart := 0 },
  { event := event3579
    frameStart := 0 },
  { event := event3580
    frameStart := 0 },
  { event := event3581
    frameStart := 0 },
  { event := event3582
    frameStart := 0 },
  { event := event3583
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events013
