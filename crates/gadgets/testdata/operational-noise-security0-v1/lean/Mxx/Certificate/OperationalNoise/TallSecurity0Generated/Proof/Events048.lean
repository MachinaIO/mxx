import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events048

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event12288 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event12289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event12290 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event12291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 12290

def event12292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 12288

def event12293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 12291 .coefficient) (.value (.predecessor 1 12292 .coefficient)))

def event12294 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event12295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 12294

def event12296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 12286

def event12297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 12295 .coefficient, .predecessor 1 12296 .coefficient])

def event12298 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event12299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 12298

def event12300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 12284

def event12301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 12300 .coefficient))

def event12302 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event12303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11401⟩⟩) 0 ⟨5560⟩ 12302

def event12304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11401⟩⟩) (.authority (.programFamilyFact))

def exact12305RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩], []⟩, (1)⟩]

theorem exact12305RawTermsValid :
    exact12305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11401⟩⟩) exact12305RawTerms (.finite 16) 12304 .exactZero (none)

def event12306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14026⟩⟩) 0 ⟨5560⟩ 12302

def event12307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14026⟩⟩) (.authority (.programFamilyFact))

def exact12308RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact12308RawTermsValid :
    exact12308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12308 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14026⟩⟩) exact12308RawTerms (.finite 16) 12307 .exactZero (none)

def event12309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 0 ⟨14026⟩ 12308

def event12310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 1 ⟨11401⟩ 12305

def event12311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.product (.predecessor 0 12309 .coefficient) (.predecessor 1 12310 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12312 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩) [⟨.result 12308 .coefficient, true, some 1⟩, ⟨.result 12305 .coefficient, true, some 1⟩])

def event12313 : Event := .survivorFold (1) 12312

def exact12314RawTerms : List Term := []

theorem exact12314RawTermsValid :
    exact12314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12314 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14027⟩⟩) exact12314RawTerms (.finite 256) 12311 (.finite 256) (some (12312))

def event12315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14028⟩⟩) 0 ⟨14027⟩ 12314

def event12316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.identity (.predecessor 0 12315 .coefficient))

def event12317 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.finite 256)

def event12318 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15837⟩⟩) 0 ⟨14028⟩ 12317

def event12319 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15837⟩⟩) (.authority (.programFamilyFact))

def exact12320RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact12320RawTermsValid :
    exact12320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12320 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15837⟩⟩) exact12320RawTerms (.finite 16) 12319 .exactZero (none)

def event12321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15838⟩⟩) 0 ⟨15837⟩ 12320

def event12322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.identity (.predecessor 0 12321 .coefficient))

def event12323 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.finite 16)

def event12324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21272⟩⟩) 0 ⟨15838⟩ 12323

def event12325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21272⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact12326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩, (1)⟩]

theorem exact12326RawTermsValid :
    exact12326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12326 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21272⟩⟩) exact12326RawTerms (.finite 136065468) 12325 .exactZero (none)

def event12327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact12328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact12328RawTermsValid :
    exact12328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact12328RawTerms .large 12327 .exactZero (none)

def event12329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21273⟩⟩) 0 ⟨6⟩ 12328

def event12330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21273⟩⟩) 1 ⟨21272⟩ 12326

def event12331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21273⟩⟩) (.product (.predecessor 0 12329 .coefficient) (.predecessor 1 12330 .coefficient) (⟨false, false, none, none, none⟩))

def event12332 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21273⟩⟩, .operator (⟨12328, 0⟩, ⟨12326, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩, (1)⟩)

def exact12333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩, (1)⟩]

theorem exact12333RawTermsValid :
    exact12333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21273⟩⟩) exact12333RawTerms .large 12331 .exactZero (none)

def event12334 : Event := .preFoldPolynomial 12333 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩, (1)⟩] .exactZero none

def exact12335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩, (1)⟩]

def event12335 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21273⟩⟩) 12334 exact12335RawTerms .large 12331 .exactZero (none)

def event12336 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27706⟩⟩)

def event12337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event12338 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event12339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event12340 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event12341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event12342 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event12343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event12344 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event12345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 12344

def event12346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 12342

def event12347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 12345 .coefficient) (.value (.predecessor 1 12346 .coefficient)))

def event12348 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event12349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 12348

def event12350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 12340

def event12351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 12349 .coefficient, .predecessor 1 12350 .coefficient])

def event12352 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event12353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 12352

def event12354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 12338

def event12355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 12354 .coefficient))

def event12356 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event12357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11401⟩⟩) 0 ⟨5560⟩ 12356

def event12358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11401⟩⟩) (.authority (.programFamilyFact))

def exact12359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩], []⟩, (1)⟩]

theorem exact12359RawTermsValid :
    exact12359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11401⟩⟩) exact12359RawTerms (.finite 16) 12358 .exactZero (none)

def event12360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14026⟩⟩) 0 ⟨5560⟩ 12356

def event12361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14026⟩⟩) (.authority (.programFamilyFact))

def exact12362RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact12362RawTermsValid :
    exact12362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12362 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14026⟩⟩) exact12362RawTerms (.finite 16) 12361 .exactZero (none)

def event12363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 0 ⟨14026⟩ 12362

def event12364 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 1 ⟨11401⟩ 12359

def event12365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.product (.predecessor 0 12363 .coefficient) (.predecessor 1 12364 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event12366 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14027⟩⟩, .operator (⟨12362, 0⟩, ⟨12359, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩)

def exact12367RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact12367RawTermsValid :
    exact12367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14027⟩⟩) exact12367RawTerms (.finite 256) 12365 .exactZero (none)

def event12368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14028⟩⟩) 0 ⟨14027⟩ 12367

def event12369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.identity (.predecessor 0 12368 .coefficient))

def event12370 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.finite 256)

def event12371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15837⟩⟩) 0 ⟨14028⟩ 12370

def event12372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15837⟩⟩) (.authority (.programFamilyFact))

def exact12373RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact12373RawTermsValid :
    exact12373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15837⟩⟩) exact12373RawTerms (.finite 16) 12372 .exactZero (none)

def event12374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15838⟩⟩) 0 ⟨15837⟩ 12373

def event12375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.identity (.predecessor 0 12374 .coefficient))

def event12376 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.finite 16)

def event12377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24109⟩⟩) 0 ⟨15838⟩ 12376

def event12378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24109⟩⟩) (.authority (.programFamilyFact))

def event12379 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24109⟩⟩) (.finite 3720)

def event12380 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event12381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24111⟩⟩) 0 ⟨6689⟩ 12380

def event12382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24111⟩⟩) 1 ⟨24109⟩ 12379

def event12383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24111⟩⟩) (.authority (.operator))

def exact12384RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩]

theorem exact12384RawTermsValid :
    exact12384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12384 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24111⟩⟩) exact12384RawTerms .large 12383 .exactZero (none)

def event12385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27701⟩⟩) 0 ⟨24111⟩ 12384

def event12386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27701⟩⟩) (.authority (.operator))

def exact12387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (1)⟩]

theorem exact12387RawTermsValid :
    exact12387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12387 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27701⟩⟩) exact12387RawTerms (.finite 8192) 12386 .exactZero (none)

def event12388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event12389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event12390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15912⟩⟩) 0 ⟨15838⟩ 12376

def event12391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15912⟩⟩) 1 ⟨110⟩ 12389

def event12392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15912⟩⟩) (.sum [.predecessor 0 12390 .coefficient, .predecessor 1 12391 .coefficient])

def event12393 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15912⟩⟩) (.finite 16)

def event12394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15913⟩⟩) 0 ⟨15912⟩ 12393

def event12395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15913⟩⟩) (.identity (.predecessor 0 12394 .coefficient))

def exact12396RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact12396RawTermsValid :
    exact12396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12396 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15913⟩⟩) exact12396RawTerms (.finite 16) 12395 .exactZero (none)

def event12397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact12398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact12398RawTermsValid :
    exact12398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12398 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact12398RawTerms .large 12397 .exactZero (none)

def event12399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15914⟩⟩) 0 ⟨6544⟩ 12398

def event12400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15914⟩⟩) 1 ⟨15913⟩ 12396

def event12401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15914⟩⟩) (.product (.predecessor 0 12399 .coefficient) (.predecessor 1 12400 .coefficient) (⟨false, false, none, none, none⟩))

def event12402 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15914⟩⟩, .operator (⟨12398, 0⟩, ⟨12396, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact12403RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact12403RawTermsValid :
    exact12403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15914⟩⟩) exact12403RawTerms .large 12401 .exactZero (none)

def event12404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 12380

def event12405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact12406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact12406RawTermsValid :
    exact12406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact12406RawTerms .large 12405 .exactZero (none)

def event12407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15915⟩⟩) 0 ⟨6696⟩ 12406

def event12408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15915⟩⟩) 1 ⟨15914⟩ 12403

def event12409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15915⟩⟩) (.sum [.predecessor 0 12407 .coefficient, .predecessor 1 12408 .coefficient])

def exact12410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12410RawTermsValid :
    exact12410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15915⟩⟩) exact12410RawTerms .large 12409 .exactZero (none)

def event12411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27702⟩⟩) 0 ⟨15915⟩ 12410

def event12412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27702⟩⟩) 1 ⟨27701⟩ 12387

def event12413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27702⟩⟩) (.product (.predecessor 0 12411 .coefficient) (.predecessor 1 12412 .coefficient) (⟨false, false, none, none, none⟩))

def event12414 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27702⟩⟩, .operator (⟨12410, 1⟩, ⟨12387, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩)

def event12415 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27702⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27701⟩⟩) ⟨24111⟩ 12384)

def event12416 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27702⟩⟩, .relation 12415 0, ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (-1)⟩)

def event12417 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27702⟩⟩, .operator (⟨12410, 0⟩, ⟨12387, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (1)⟩)

def exact12418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (-1)⟩]

theorem exact12418RawTermsValid :
    exact12418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27702⟩⟩) exact12418RawTerms .large 12413 .exactZero (none)

def event12419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15879⟩⟩) 0 ⟨15838⟩ 12376

def event12420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15879⟩⟩) (.authority (.programFamilyFact))

def exact12421RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], []⟩, (1)⟩]

theorem exact12421RawTermsValid :
    exact12421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12421 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15879⟩⟩) exact12421RawTerms (.finite 60) 12420 .exactZero (none)

def event12422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15880⟩⟩) 0 ⟨6544⟩ 12398

def event12423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15880⟩⟩) 1 ⟨15879⟩ 12421

def event12424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15880⟩⟩) (.product (.predecessor 0 12422 .coefficient) (.predecessor 1 12423 .coefficient) (⟨false, true, none, none, some 1⟩))

def event12425 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15880⟩⟩, .operator (⟨12398, 0⟩, ⟨12421, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact12426RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact12426RawTermsValid :
    exact12426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15880⟩⟩) exact12426RawTerms .large 12424 .exactZero (none)

def event12427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6721⟩⟩) 0 ⟨6689⟩ 12380

def event12428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6721⟩⟩) (.authority (.operator))

def exact12429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩]

theorem exact12429RawTermsValid :
    exact12429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6721⟩⟩) exact12429RawTerms .large 12428 .exactZero (none)

def event12430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15881⟩⟩) 0 ⟨6721⟩ 12429

def event12431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15881⟩⟩) 1 ⟨15880⟩ 12426

def event12432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15881⟩⟩) (.sum [.predecessor 0 12430 .coefficient, .predecessor 1 12431 .coefficient])

def exact12433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12433RawTermsValid :
    exact12433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12433 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15881⟩⟩) exact12433RawTerms .large 12432 .exactZero (none)

def event12434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27706⟩⟩) 0 ⟨15881⟩ 12433

def event12435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27706⟩⟩) 1 ⟨27702⟩ 12418

def event12436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27706⟩⟩) (.sum [.predecessor 0 12434 .coefficient, .predecessor 1 12435 .coefficient])

def exact12437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12437RawTermsValid :
    exact12437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12437 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27706⟩⟩) exact12437RawTerms .large 12436 .exactZero (none)

def event12438 : Event := .preFoldPolynomial 12437 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact12439RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event12439 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27706⟩⟩) 12438 exact12439RawTerms .large 12436 .exactZero (none)

def event12440 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15838⟩⟩) ⟨⟨134⟩, ⟨41⟩, ⟨109⟩⟩ ⟨12282, 12440⟩

def event12441 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21275⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩) (1) 0 2 (.universal 12440 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩) (none) 12439)

def event12442 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21275⟩⟩, .relation 12441 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩)

def event12443 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21275⟩⟩, .relation 12441 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩)

def event12444 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21275⟩⟩, .relation 12441 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event12445 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21275⟩⟩, .relation 12441 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩)

def exact12446RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12446RawTermsValid :
    exact12446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21275⟩⟩) exact12446RawTerms .large 12278 (.finite 1811303510016) (some (12280))

def event12447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27704⟩⟩) 0 ⟨21275⟩ 12446

def event12448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27704⟩⟩) 1 ⟨27703⟩ 12268

def event12449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27704⟩⟩) (.sum [.predecessor 0 12447 .coefficient, .predecessor 1 12448 .coefficient])

def event12450 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27704⟩⟩, .operator (⟨12446, 2⟩, ⟨12268, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24111⟩⟩]⟩, (-1)⟩)

def event12451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27704⟩⟩, .operator (⟨12446, 0⟩, ⟨12268, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩, (1)⟩)

def event12452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27704⟩⟩) (.sum [.result 12446 .summary, .result 12268 .summary])

def exact12453RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12453RawTermsValid :
    exact12453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27704⟩⟩) exact12453RawTerms .large 12449 (.finite 1292046061494565744640) (some (12452))

def event12454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24046⟩⟩) 0 ⟨15719⟩ 344

def event12455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24046⟩⟩) (.authority (.programFamilyFact))

def event12456 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24046⟩⟩) (.finite 3720)

def event12457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24048⟩⟩) 0 ⟨6689⟩ 5477

def event12458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24048⟩⟩) 1 ⟨24046⟩ 12456

def event12459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24048⟩⟩) (.authority (.operator))

def exact12460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24048⟩⟩]⟩, (1)⟩]

theorem exact12460RawTermsValid :
    exact12460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12460 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24048⟩⟩) exact12460RawTerms .large 12459 .exactZero (none)

def event12461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27484⟩⟩) 0 ⟨24048⟩ 12460

def event12462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27484⟩⟩) (.authority (.operator))

def exact12463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27484⟩⟩]⟩, (1)⟩]

theorem exact12463RawTermsValid :
    exact12463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12463 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27484⟩⟩) exact12463RawTerms (.finite 8192) 12462 .exactZero (none)

def event12464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23507⟩⟩) 0 ⟨13811⟩ 338

def event12465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23507⟩⟩) (.authority (.programFamilyFact))

def event12466 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23507⟩⟩) (.finite 3720)

def event12467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23508⟩⟩) 0 ⟨6689⟩ 5477

def event12468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23508⟩⟩) 1 ⟨23507⟩ 12466

def event12469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23508⟩⟩) (.authority (.operator))

def exact12470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23508⟩⟩]⟩, (1)⟩]

theorem exact12470RawTermsValid :
    exact12470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12470 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23508⟩⟩) exact12470RawTerms .large 12469 .exactZero (none)

def event12471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25932⟩⟩) 0 ⟨23508⟩ 12470

def event12472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25932⟩⟩) (.authority (.operator))

def exact12473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25932⟩⟩]⟩, (1)⟩]

theorem exact12473RawTermsValid :
    exact12473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12473 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25932⟩⟩) exact12473RawTerms (.finite 8192) 12472 .exactZero (none)

def event12474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨91⟩⟩) 0 ⟨11⟩ 6441

def event12475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨91⟩⟩) (.identity (.predecessor 0 12474 .coefficient))

def exact12476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩, (1)⟩]

theorem exact12476RawTermsValid :
    exact12476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨91⟩⟩) exact12476RawTerms (.finite 26) 12475 .exactZero (none)

def event12477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11318⟩⟩) 0 ⟨11317⟩ 327

def event12478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11318⟩⟩) 1 ⟨6571⟩ 6449

def event12479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11318⟩⟩) (.tensor (.predecessor 0 12477 .coefficient) (.predecessor 1 12478 .coefficient) true false)

def event12480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11318⟩⟩, .operator (⟨327, 0⟩, ⟨6449, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact12481RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact12481RawTermsValid :
    exact12481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11318⟩⟩) exact12481RawTerms .large 12479 .exactZero (none)

def event12482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6777⟩⟩) 0 ⟨6757⟩ 5870

def event12483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6777⟩⟩) (.identity (.predecessor 0 12482 .coefficient))

def exact12484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact12484RawTermsValid :
    exact12484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6777⟩⟩) exact12484RawTerms .large 12483 .exactZero (none)

def event12485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7385⟩⟩) 0 ⟨5563⟩ 6314

def event12486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7385⟩⟩) 1 ⟨6777⟩ 12484

def event12487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7385⟩⟩) (.product (.predecessor 0 12485 .coefficient) (.predecessor 1 12486 .coefficient) (⟨false, false, none, none, none⟩))

def event12488 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7385⟩⟩, .operator (⟨6314, 0⟩, ⟨12484, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact12489RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact12489RawTermsValid :
    exact12489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12489 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7385⟩⟩) exact12489RawTerms .large 12487 .exactZero (none)

def event12490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11319⟩⟩) 0 ⟨7385⟩ 12489

def event12491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11319⟩⟩) 1 ⟨11318⟩ 12481

def event12492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11319⟩⟩) (.sum [.predecessor 0 12490 .coefficient, .predecessor 1 12491 .coefficient])

def exact12493RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12493RawTermsValid :
    exact12493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12493 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11319⟩⟩) exact12493RawTerms .large 12492 .exactZero (none)

def event12494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11320⟩⟩) 0 ⟨11319⟩ 12493

def event12495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11320⟩⟩) 1 ⟨91⟩ 12476

def event12496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11320⟩⟩) (.sum [.predecessor 0 12494 .coefficient, .predecessor 1 12495 .coefficient])

def event12497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11320⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) [⟨.result 12476 .coefficient, false, none⟩])

def event12498 : Event := .survivorFold (1) 12497

def exact12499RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12499RawTermsValid :
    exact12499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12499 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11320⟩⟩) exact12499RawTerms .large 12496 (.finite 26) (some (12497))

def event12500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13812⟩⟩) 0 ⟨11320⟩ 12499

def event12501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13812⟩⟩) 1 ⟨13809⟩ 330

def event12502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13812⟩⟩) (.product (.predecessor 0 12500 .coefficient) (.predecessor 1 12501 .coefficient) (⟨false, true, none, none, some 1⟩))

def event12503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13812⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13809⟩⟩], []⟩) [⟨.result 330 .coefficient, true, some 1⟩])

def event12504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13812⟩⟩) (.product (.result 12499 .summary) (.transfer 12503) (⟨false, false, none, none, none⟩))

def event12505 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13812⟩⟩, .operator (⟨12499, 1⟩, ⟨330, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event12506 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13812⟩⟩, .operator (⟨12499, 0⟩, ⟨330, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact12507RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11317⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact12507RawTermsValid :
    exact12507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12507 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13812⟩⟩) exact12507RawTerms .large 12502 (.finite 9984) (some (12504))

def event12508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7846⟩⟩) 0 ⟨6777⟩ 12484

def event12509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7846⟩⟩) (.authority (.operator))

def exact12510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact12510RawTermsValid :
    exact12510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12510 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7846⟩⟩) exact12510RawTerms (.finite 8192) 12509 .exactZero (none)

def event12511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 0 ⟨7846⟩ 12510

def event12512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 1 ⟨2348⟩ 4

def event12513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7847⟩⟩) (.scale (.predecessor 0 12511 .coefficient) (.value (.predecessor 1 12512 .coefficient)))

def exact12514RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact12514RawTermsValid :
    exact12514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12514 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7847⟩⟩) exact12514RawTerms (.finite 8192) 12513 .exactZero (none)

def event12515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨108⟩⟩) 0 ⟨11⟩ 6441

def event12516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨108⟩⟩) (.identity (.predecessor 0 12515 .coefficient))

def exact12517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩, (1)⟩]

theorem exact12517RawTermsValid :
    exact12517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨108⟩⟩) exact12517RawTerms (.finite 26) 12516 .exactZero (none)

def event12518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13813⟩⟩) 0 ⟨13809⟩ 330

def event12519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13813⟩⟩) 1 ⟨6571⟩ 6449

def event12520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13813⟩⟩) (.tensor (.predecessor 0 12518 .coefficient) (.predecessor 1 12519 .coefficient) true false)

def event12521 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13813⟩⟩, .operator (⟨330, 0⟩, ⟨6449, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact12522RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact12522RawTermsValid :
    exact12522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13813⟩⟩) exact12522RawTerms .large 12520 .exactZero (none)

def event12523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6794⟩⟩) 0 ⟨6757⟩ 5870

def event12524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6794⟩⟩) (.identity (.predecessor 0 12523 .coefficient))

def exact12525RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact12525RawTermsValid :
    exact12525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6794⟩⟩) exact12525RawTerms .large 12524 .exactZero (none)

def event12526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7402⟩⟩) 0 ⟨5563⟩ 6314

def event12527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7402⟩⟩) 1 ⟨6794⟩ 12525

def event12528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7402⟩⟩) (.product (.predecessor 0 12526 .coefficient) (.predecessor 1 12527 .coefficient) (⟨false, false, none, none, none⟩))

def event12529 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7402⟩⟩, .operator (⟨6314, 0⟩, ⟨12525, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩)

def exact12530RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact12530RawTermsValid :
    exact12530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7402⟩⟩) exact12530RawTerms .large 12528 .exactZero (none)

def event12531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13814⟩⟩) 0 ⟨7402⟩ 12530

def event12532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13814⟩⟩) 1 ⟨13813⟩ 12522

def event12533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13814⟩⟩) (.sum [.predecessor 0 12531 .coefficient, .predecessor 1 12532 .coefficient])

def exact12534RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12534RawTermsValid :
    exact12534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13814⟩⟩) exact12534RawTerms .large 12533 .exactZero (none)

def event12535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13815⟩⟩) 0 ⟨13814⟩ 12534

def event12536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13815⟩⟩) 1 ⟨108⟩ 12517

def event12537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13815⟩⟩) (.sum [.predecessor 0 12535 .coefficient, .predecessor 1 12536 .coefficient])

def event12538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13815⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩) [⟨.result 12517 .coefficient, false, none⟩])

def event12539 : Event := .survivorFold (1) 12538

def exact12540RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact12540RawTermsValid :
    exact12540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event12540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13815⟩⟩) exact12540RawTerms .large 12537 (.finite 26) (some (12538))

def event12541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13816⟩⟩) 0 ⟨13815⟩ 12540

def event12542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13816⟩⟩) 1 ⟨7847⟩ 12514

def event12543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13816⟩⟩) (.product (.predecessor 0 12541 .coefficient) (.predecessor 1 12542 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf768 : Array AnnotatedEvent := #[
  { event := event12288
    frameStart := 12282 },
  { event := event12289
    frameStart := 12282 },
  { event := event12290
    frameStart := 12282 },
  { event := event12291
    frameStart := 12282 },
  { event := event12292
    frameStart := 12282 },
  { event := event12293
    frameStart := 12282 },
  { event := event12294
    frameStart := 12282 },
  { event := event12295
    frameStart := 12282 },
  { event := event12296
    frameStart := 12282 },
  { event := event12297
    frameStart := 12282 },
  { event := event12298
    frameStart := 12282 },
  { event := event12299
    frameStart := 12282 },
  { event := event12300
    frameStart := 12282 },
  { event := event12301
    frameStart := 12282 },
  { event := event12302
    frameStart := 12282 },
  { event := event12303
    frameStart := 12282 }
]

def eventLeaf769 : Array AnnotatedEvent := #[
  { event := event12304
    frameStart := 12282 },
  { event := event12305
    frameStart := 12282 },
  { event := event12306
    frameStart := 12282 },
  { event := event12307
    frameStart := 12282 },
  { event := event12308
    frameStart := 12282 },
  { event := event12309
    frameStart := 12282 },
  { event := event12310
    frameStart := 12282 },
  { event := event12311
    frameStart := 12282 },
  { event := event12312
    frameStart := 12282 },
  { event := event12313
    frameStart := 12282 },
  { event := event12314
    frameStart := 12282 },
  { event := event12315
    frameStart := 12282 },
  { event := event12316
    frameStart := 12282 },
  { event := event12317
    frameStart := 12282 },
  { event := event12318
    frameStart := 12282 },
  { event := event12319
    frameStart := 12282 }
]

def eventLeaf770 : Array AnnotatedEvent := #[
  { event := event12320
    frameStart := 12282 },
  { event := event12321
    frameStart := 12282 },
  { event := event12322
    frameStart := 12282 },
  { event := event12323
    frameStart := 12282 },
  { event := event12324
    frameStart := 12282 },
  { event := event12325
    frameStart := 12282 },
  { event := event12326
    frameStart := 12282 },
  { event := event12327
    frameStart := 12282 },
  { event := event12328
    frameStart := 12282 },
  { event := event12329
    frameStart := 12282 },
  { event := event12330
    frameStart := 12282 },
  { event := event12331
    frameStart := 12282 },
  { event := event12332
    frameStart := 12282 },
  { event := event12333
    frameStart := 12282 },
  { event := event12334
    frameStart := 12282 },
  { event := event12335
    frameStart := 12282 }
]

def eventLeaf771 : Array AnnotatedEvent := #[
  { event := event12336
    frameStart := 12336 },
  { event := event12337
    frameStart := 12336 },
  { event := event12338
    frameStart := 12336 },
  { event := event12339
    frameStart := 12336 },
  { event := event12340
    frameStart := 12336 },
  { event := event12341
    frameStart := 12336 },
  { event := event12342
    frameStart := 12336 },
  { event := event12343
    frameStart := 12336 },
  { event := event12344
    frameStart := 12336 },
  { event := event12345
    frameStart := 12336 },
  { event := event12346
    frameStart := 12336 },
  { event := event12347
    frameStart := 12336 },
  { event := event12348
    frameStart := 12336 },
  { event := event12349
    frameStart := 12336 },
  { event := event12350
    frameStart := 12336 },
  { event := event12351
    frameStart := 12336 }
]

def eventLeaf772 : Array AnnotatedEvent := #[
  { event := event12352
    frameStart := 12336 },
  { event := event12353
    frameStart := 12336 },
  { event := event12354
    frameStart := 12336 },
  { event := event12355
    frameStart := 12336 },
  { event := event12356
    frameStart := 12336 },
  { event := event12357
    frameStart := 12336 },
  { event := event12358
    frameStart := 12336 },
  { event := event12359
    frameStart := 12336 },
  { event := event12360
    frameStart := 12336 },
  { event := event12361
    frameStart := 12336 },
  { event := event12362
    frameStart := 12336 },
  { event := event12363
    frameStart := 12336 },
  { event := event12364
    frameStart := 12336 },
  { event := event12365
    frameStart := 12336 },
  { event := event12366
    frameStart := 12336 },
  { event := event12367
    frameStart := 12336 }
]

def eventLeaf773 : Array AnnotatedEvent := #[
  { event := event12368
    frameStart := 12336 },
  { event := event12369
    frameStart := 12336 },
  { event := event12370
    frameStart := 12336 },
  { event := event12371
    frameStart := 12336 },
  { event := event12372
    frameStart := 12336 },
  { event := event12373
    frameStart := 12336 },
  { event := event12374
    frameStart := 12336 },
  { event := event12375
    frameStart := 12336 },
  { event := event12376
    frameStart := 12336 },
  { event := event12377
    frameStart := 12336 },
  { event := event12378
    frameStart := 12336 },
  { event := event12379
    frameStart := 12336 },
  { event := event12380
    frameStart := 12336 },
  { event := event12381
    frameStart := 12336 },
  { event := event12382
    frameStart := 12336 },
  { event := event12383
    frameStart := 12336 }
]

def eventLeaf774 : Array AnnotatedEvent := #[
  { event := event12384
    frameStart := 12336 },
  { event := event12385
    frameStart := 12336 },
  { event := event12386
    frameStart := 12336 },
  { event := event12387
    frameStart := 12336 },
  { event := event12388
    frameStart := 12336 },
  { event := event12389
    frameStart := 12336 },
  { event := event12390
    frameStart := 12336 },
  { event := event12391
    frameStart := 12336 },
  { event := event12392
    frameStart := 12336 },
  { event := event12393
    frameStart := 12336 },
  { event := event12394
    frameStart := 12336 },
  { event := event12395
    frameStart := 12336 },
  { event := event12396
    frameStart := 12336 },
  { event := event12397
    frameStart := 12336 },
  { event := event12398
    frameStart := 12336 },
  { event := event12399
    frameStart := 12336 }
]

def eventLeaf775 : Array AnnotatedEvent := #[
  { event := event12400
    frameStart := 12336 },
  { event := event12401
    frameStart := 12336 },
  { event := event12402
    frameStart := 12336 },
  { event := event12403
    frameStart := 12336 },
  { event := event12404
    frameStart := 12336 },
  { event := event12405
    frameStart := 12336 },
  { event := event12406
    frameStart := 12336 },
  { event := event12407
    frameStart := 12336 },
  { event := event12408
    frameStart := 12336 },
  { event := event12409
    frameStart := 12336 },
  { event := event12410
    frameStart := 12336 },
  { event := event12411
    frameStart := 12336 },
  { event := event12412
    frameStart := 12336 },
  { event := event12413
    frameStart := 12336 },
  { event := event12414
    frameStart := 12336 },
  { event := event12415
    frameStart := 12336 }
]

def eventLeaf776 : Array AnnotatedEvent := #[
  { event := event12416
    frameStart := 12336 },
  { event := event12417
    frameStart := 12336 },
  { event := event12418
    frameStart := 12336 },
  { event := event12419
    frameStart := 12336 },
  { event := event12420
    frameStart := 12336 },
  { event := event12421
    frameStart := 12336 },
  { event := event12422
    frameStart := 12336 },
  { event := event12423
    frameStart := 12336 },
  { event := event12424
    frameStart := 12336 },
  { event := event12425
    frameStart := 12336 },
  { event := event12426
    frameStart := 12336 },
  { event := event12427
    frameStart := 12336 },
  { event := event12428
    frameStart := 12336 },
  { event := event12429
    frameStart := 12336 },
  { event := event12430
    frameStart := 12336 },
  { event := event12431
    frameStart := 12336 }
]

def eventLeaf777 : Array AnnotatedEvent := #[
  { event := event12432
    frameStart := 12336 },
  { event := event12433
    frameStart := 12336 },
  { event := event12434
    frameStart := 12336 },
  { event := event12435
    frameStart := 12336 },
  { event := event12436
    frameStart := 12336 },
  { event := event12437
    frameStart := 12336 },
  { event := event12438
    frameStart := 12336 },
  { event := event12439
    frameStart := 12336 },
  { event := event12440
    frameStart := 0 },
  { event := event12441
    frameStart := 0 },
  { event := event12442
    frameStart := 0 },
  { event := event12443
    frameStart := 0 },
  { event := event12444
    frameStart := 0 },
  { event := event12445
    frameStart := 0 },
  { event := event12446
    frameStart := 0 },
  { event := event12447
    frameStart := 0 }
]

def eventLeaf778 : Array AnnotatedEvent := #[
  { event := event12448
    frameStart := 0 },
  { event := event12449
    frameStart := 0 },
  { event := event12450
    frameStart := 0 },
  { event := event12451
    frameStart := 0 },
  { event := event12452
    frameStart := 0 },
  { event := event12453
    frameStart := 0 },
  { event := event12454
    frameStart := 0 },
  { event := event12455
    frameStart := 0 },
  { event := event12456
    frameStart := 0 },
  { event := event12457
    frameStart := 0 },
  { event := event12458
    frameStart := 0 },
  { event := event12459
    frameStart := 0 },
  { event := event12460
    frameStart := 0 },
  { event := event12461
    frameStart := 0 },
  { event := event12462
    frameStart := 0 },
  { event := event12463
    frameStart := 0 }
]

def eventLeaf779 : Array AnnotatedEvent := #[
  { event := event12464
    frameStart := 0 },
  { event := event12465
    frameStart := 0 },
  { event := event12466
    frameStart := 0 },
  { event := event12467
    frameStart := 0 },
  { event := event12468
    frameStart := 0 },
  { event := event12469
    frameStart := 0 },
  { event := event12470
    frameStart := 0 },
  { event := event12471
    frameStart := 0 },
  { event := event12472
    frameStart := 0 },
  { event := event12473
    frameStart := 0 },
  { event := event12474
    frameStart := 0 },
  { event := event12475
    frameStart := 0 },
  { event := event12476
    frameStart := 0 },
  { event := event12477
    frameStart := 0 },
  { event := event12478
    frameStart := 0 },
  { event := event12479
    frameStart := 0 }
]

def eventLeaf780 : Array AnnotatedEvent := #[
  { event := event12480
    frameStart := 0 },
  { event := event12481
    frameStart := 0 },
  { event := event12482
    frameStart := 0 },
  { event := event12483
    frameStart := 0 },
  { event := event12484
    frameStart := 0 },
  { event := event12485
    frameStart := 0 },
  { event := event12486
    frameStart := 0 },
  { event := event12487
    frameStart := 0 },
  { event := event12488
    frameStart := 0 },
  { event := event12489
    frameStart := 0 },
  { event := event12490
    frameStart := 0 },
  { event := event12491
    frameStart := 0 },
  { event := event12492
    frameStart := 0 },
  { event := event12493
    frameStart := 0 },
  { event := event12494
    frameStart := 0 },
  { event := event12495
    frameStart := 0 }
]

def eventLeaf781 : Array AnnotatedEvent := #[
  { event := event12496
    frameStart := 0 },
  { event := event12497
    frameStart := 0 },
  { event := event12498
    frameStart := 0 },
  { event := event12499
    frameStart := 0 },
  { event := event12500
    frameStart := 0 },
  { event := event12501
    frameStart := 0 },
  { event := event12502
    frameStart := 0 },
  { event := event12503
    frameStart := 0 },
  { event := event12504
    frameStart := 0 },
  { event := event12505
    frameStart := 0 },
  { event := event12506
    frameStart := 0 },
  { event := event12507
    frameStart := 0 },
  { event := event12508
    frameStart := 0 },
  { event := event12509
    frameStart := 0 },
  { event := event12510
    frameStart := 0 },
  { event := event12511
    frameStart := 0 }
]

def eventLeaf782 : Array AnnotatedEvent := #[
  { event := event12512
    frameStart := 0 },
  { event := event12513
    frameStart := 0 },
  { event := event12514
    frameStart := 0 },
  { event := event12515
    frameStart := 0 },
  { event := event12516
    frameStart := 0 },
  { event := event12517
    frameStart := 0 },
  { event := event12518
    frameStart := 0 },
  { event := event12519
    frameStart := 0 },
  { event := event12520
    frameStart := 0 },
  { event := event12521
    frameStart := 0 },
  { event := event12522
    frameStart := 0 },
  { event := event12523
    frameStart := 0 },
  { event := event12524
    frameStart := 0 },
  { event := event12525
    frameStart := 0 },
  { event := event12526
    frameStart := 0 },
  { event := event12527
    frameStart := 0 }
]

def eventLeaf783 : Array AnnotatedEvent := #[
  { event := event12528
    frameStart := 0 },
  { event := event12529
    frameStart := 0 },
  { event := event12530
    frameStart := 0 },
  { event := event12531
    frameStart := 0 },
  { event := event12532
    frameStart := 0 },
  { event := event12533
    frameStart := 0 },
  { event := event12534
    frameStart := 0 },
  { event := event12535
    frameStart := 0 },
  { event := event12536
    frameStart := 0 },
  { event := event12537
    frameStart := 0 },
  { event := event12538
    frameStart := 0 },
  { event := event12539
    frameStart := 0 },
  { event := event12540
    frameStart := 0 },
  { event := event12541
    frameStart := 0 },
  { event := event12542
    frameStart := 0 },
  { event := event12543
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events048
