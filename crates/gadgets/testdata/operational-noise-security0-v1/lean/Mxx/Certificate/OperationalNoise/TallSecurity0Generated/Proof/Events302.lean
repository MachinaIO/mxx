import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events302

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact77312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩]

theorem exact77312RawTermsValid :
    exact77312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21614⟩⟩) exact77312RawTerms (.finite 136065468) 77311 .exactZero (none)

def event77313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21615⟩⟩) 0 ⟨5535⟩ 65387

def event77314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21615⟩⟩) 1 ⟨21614⟩ 77312

def event77315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21615⟩⟩) (.product (.predecessor 0 77313 .coefficient) (.predecessor 1 77314 .coefficient) (⟨false, false, none, none, none⟩))

def event77316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21615⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩) [⟨.result 77308 .coefficient, false, none⟩])

def event77317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21615⟩⟩) (.product (.result 65387 .summary) (.transfer 77316) (⟨false, false, none, none, none⟩))

def event77318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21615⟩⟩, .operator (⟨65387, 0⟩, ⟨77312, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩)

def event77319 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21613⟩⟩)

def event77320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77321 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77323 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77325 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77327 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77327

def event77329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77325

def event77330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77328 .coefficient) (.value (.predecessor 1 77329 .coefficient)))

def event77331 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77331

def event77333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77323

def event77334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77332 .coefficient, .predecessor 1 77333 .coefficient])

def event77335 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77335

def event77337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77321

def event77338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77337 .coefficient))

def event77339 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 77339

def event77341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact77342RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact77342RawTermsValid :
    exact77342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77342 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact77342RawTerms (.finite 28) 77341 .exactZero (none)

def event77343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 77339

def event77344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact77345RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact77345RawTermsValid :
    exact77345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77345 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact77345RawTerms (.finite 28) 77344 .exactZero (none)

def event77346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 77345

def event77347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 77342

def event77348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 77346 .coefficient) (.predecessor 1 77347 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩) [⟨.result 77345 .coefficient, true, some 1⟩, ⟨.result 77342 .coefficient, true, some 1⟩])

def event77350 : Event := .survivorFold (1) 77349

def exact77351RawTerms : List Term := []

theorem exact77351RawTermsValid :
    exact77351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact77351RawTerms (.finite 784) 77348 (.finite 784) (some (77349))

def event77352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 77351

def event77353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 77352 .coefficient))

def event77354 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event77355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16174⟩⟩) 0 ⟨14634⟩ 77354

def event77356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16174⟩⟩) (.authority (.programFamilyFact))

def exact77357RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact77357RawTermsValid :
    exact77357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77357 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16174⟩⟩) exact77357RawTerms (.finite 28) 77356 .exactZero (none)

def event77358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16175⟩⟩) 0 ⟨16174⟩ 77357

def event77359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.identity (.predecessor 0 77358 .coefficient))

def event77360 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.finite 28)

def event77361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21612⟩⟩) 0 ⟨16175⟩ 77360

def event77362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21612⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact77363RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩]

theorem exact77363RawTermsValid :
    exact77363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77363 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21612⟩⟩) exact77363RawTerms (.finite 136065468) 77362 .exactZero (none)

def event77364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact77365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact77365RawTermsValid :
    exact77365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact77365RawTerms .large 77364 .exactZero (none)

def event77366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21613⟩⟩) 0 ⟨6⟩ 77365

def event77367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21613⟩⟩) 1 ⟨21612⟩ 77363

def event77368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21613⟩⟩) (.product (.predecessor 0 77366 .coefficient) (.predecessor 1 77367 .coefficient) (⟨false, false, none, none, none⟩))

def event77369 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21613⟩⟩, .operator (⟨77365, 0⟩, ⟨77363, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩)

def exact77370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩]

theorem exact77370RawTermsValid :
    exact77370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21613⟩⟩) exact77370RawTerms .large 77368 .exactZero (none)

def event77371 : Event := .preFoldPolynomial 77370 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩] .exactZero none

def exact77372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩, (1)⟩]

def event77372 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21613⟩⟩) 77371 exact77372RawTerms .large 77368 .exactZero (none)

def event77373 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28286⟩⟩)

def event77374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77375 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77377 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77379 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77381 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77381

def event77383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77379

def event77384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77382 .coefficient) (.value (.predecessor 1 77383 .coefficient)))

def event77385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77385

def event77387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77377

def event77388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77386 .coefficient, .predecessor 1 77387 .coefficient])

def event77389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77389

def event77391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77375

def event77392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77391 .coefficient))

def event77393 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 77393

def event77395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact77396RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact77396RawTermsValid :
    exact77396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77396 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact77396RawTerms (.finite 28) 77395 .exactZero (none)

def event77397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 77393

def event77398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact77399RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact77399RawTermsValid :
    exact77399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77399 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact77399RawTerms (.finite 28) 77398 .exactZero (none)

def event77400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 77399

def event77401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 77396

def event77402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 77400 .coefficient) (.predecessor 1 77401 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77403 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14633⟩⟩, .operator (⟨77399, 0⟩, ⟨77396, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩)

def exact77404RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact77404RawTermsValid :
    exact77404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact77404RawTerms (.finite 784) 77402 .exactZero (none)

def event77405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 77404

def event77406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 77405 .coefficient))

def event77407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event77408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16174⟩⟩) 0 ⟨14634⟩ 77407

def event77409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16174⟩⟩) (.authority (.programFamilyFact))

def exact77410RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact77410RawTermsValid :
    exact77410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16174⟩⟩) exact77410RawTerms (.finite 28) 77409 .exactZero (none)

def event77411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16175⟩⟩) 0 ⟨16174⟩ 77410

def event77412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.identity (.predecessor 0 77411 .coefficient))

def event77413 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.finite 28)

def event77414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24283⟩⟩) 0 ⟨16175⟩ 77413

def event77415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.authority (.programFamilyFact))

def event77416 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.finite 3720)

def event77417 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event77418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24284⟩⟩) 0 ⟨6689⟩ 77417

def event77419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24284⟩⟩) 1 ⟨24283⟩ 77416

def event77420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24284⟩⟩) (.authority (.operator))

def exact77421RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩]

theorem exact77421RawTermsValid :
    exact77421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77421 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24284⟩⟩) exact77421RawTerms .large 77420 .exactZero (none)

def event77422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28280⟩⟩) 0 ⟨24284⟩ 77421

def event77423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28280⟩⟩) (.authority (.operator))

def exact77424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (1)⟩]

theorem exact77424RawTermsValid :
    exact77424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28280⟩⟩) exact77424RawTerms (.finite 8192) 77423 .exactZero (none)

def event77425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event77426 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event77427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16214⟩⟩) 0 ⟨16175⟩ 77413

def event77428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16214⟩⟩) 1 ⟨110⟩ 77426

def event77429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16214⟩⟩) (.sum [.predecessor 0 77427 .coefficient, .predecessor 1 77428 .coefficient])

def event77430 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16214⟩⟩) (.finite 28)

def event77431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16215⟩⟩) 0 ⟨16214⟩ 77430

def event77432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16215⟩⟩) (.identity (.predecessor 0 77431 .coefficient))

def exact77433RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact77433RawTermsValid :
    exact77433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77433 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16215⟩⟩) exact77433RawTerms (.finite 28) 77432 .exactZero (none)

def event77434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact77435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77435RawTermsValid :
    exact77435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77435 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact77435RawTerms .large 77434 .exactZero (none)

def event77436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16216⟩⟩) 0 ⟨6544⟩ 77435

def event77437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16216⟩⟩) 1 ⟨16215⟩ 77433

def event77438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16216⟩⟩) (.product (.predecessor 0 77436 .coefficient) (.predecessor 1 77437 .coefficient) (⟨false, false, none, none, none⟩))

def event77439 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16216⟩⟩, .operator (⟨77435, 0⟩, ⟨77433, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77440RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77440RawTermsValid :
    exact77440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77440 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16216⟩⟩) exact77440RawTerms .large 77438 .exactZero (none)

def event77441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 77417

def event77442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact77443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact77443RawTermsValid :
    exact77443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact77443RawTerms .large 77442 .exactZero (none)

def event77444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16217⟩⟩) 0 ⟨6699⟩ 77443

def event77445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16217⟩⟩) 1 ⟨16216⟩ 77440

def event77446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16217⟩⟩) (.sum [.predecessor 0 77444 .coefficient, .predecessor 1 77445 .coefficient])

def exact77447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77447RawTermsValid :
    exact77447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77447 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16217⟩⟩) exact77447RawTerms .large 77446 .exactZero (none)

def event77448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28281⟩⟩) 0 ⟨16217⟩ 77447

def event77449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28281⟩⟩) 1 ⟨28280⟩ 77424

def event77450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28281⟩⟩) (.product (.predecessor 0 77448 .coefficient) (.predecessor 1 77449 .coefficient) (⟨false, false, none, none, none⟩))

def event77451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28281⟩⟩, .operator (⟨77447, 0⟩, ⟨77424, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (1)⟩)

def event77452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28281⟩⟩, .operator (⟨77447, 1⟩, ⟨77424, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩)

def event77453 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28281⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28280⟩⟩) ⟨24284⟩ 77421)

def event77454 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28281⟩⟩, .relation 77453 0, ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (-1)⟩)

def exact77455RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (-1)⟩]

theorem exact77455RawTermsValid :
    exact77455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77455 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28281⟩⟩) exact77455RawTerms .large 77450 .exactZero (none)

def event77456 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17658⟩⟩) 0 ⟨16175⟩ 77413

def event77457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17658⟩⟩) (.authority (.programFamilyFact))

def exact77458RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩, (1)⟩]

theorem exact77458RawTermsValid :
    exact77458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77458 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17658⟩⟩) exact77458RawTerms (.finite 28) 77457 .exactZero (none)

def event77459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17660⟩⟩) 0 ⟨6544⟩ 77435

def event77460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17660⟩⟩) 1 ⟨17658⟩ 77458

def event77461 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17660⟩⟩) (.product (.predecessor 0 77459 .coefficient) (.predecessor 1 77460 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77462 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17660⟩⟩, .operator (⟨77435, 0⟩, ⟨77458, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77463RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77463RawTermsValid :
    exact77463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77463 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17660⟩⟩) exact77463RawTerms .large 77461 .exactZero (none)

def event77464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6726⟩⟩) 0 ⟨6689⟩ 77417

def event77465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6726⟩⟩) (.authority (.operator))

def exact77466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩]

theorem exact77466RawTermsValid :
    exact77466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77466 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6726⟩⟩) exact77466RawTerms .large 77465 .exactZero (none)

def event77467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17661⟩⟩) 0 ⟨6726⟩ 77466

def event77468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17661⟩⟩) 1 ⟨17660⟩ 77463

def event77469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17661⟩⟩) (.sum [.predecessor 0 77467 .coefficient, .predecessor 1 77468 .coefficient])

def exact77470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77470RawTermsValid :
    exact77470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77470 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17661⟩⟩) exact77470RawTerms .large 77469 .exactZero (none)

def event77471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28286⟩⟩) 0 ⟨17661⟩ 77470

def event77472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28286⟩⟩) 1 ⟨28281⟩ 77455

def event77473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28286⟩⟩) (.sum [.predecessor 0 77471 .coefficient, .predecessor 1 77472 .coefficient])

def exact77474RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77474RawTermsValid :
    exact77474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28286⟩⟩) exact77474RawTerms .large 77473 .exactZero (none)

def event77475 : Event := .preFoldPolynomial 77474 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event77476 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28286⟩⟩) 77475 exact77476RawTerms .large 77473 .exactZero (none)

def event77477 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16175⟩⟩) ⟨⟨139⟩, ⟨47⟩, ⟨109⟩⟩ ⟨77319, 77477⟩

def event77478 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21615⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩) (1) 0 2 (.universal 77477 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21612⟩⟩]⟩) (none) 77476)

def event77479 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21615⟩⟩, .relation 77478 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩)

def event77480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21615⟩⟩, .relation 77478 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩)

def event77481 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21615⟩⟩, .relation 77478 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩)

def event77482 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21615⟩⟩, .relation 77478 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77483RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77483RawTermsValid :
    exact77483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21615⟩⟩) exact77483RawTerms .large 77315 (.finite 1811303510016) (some (77317))

def event77484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28283⟩⟩) 0 ⟨21615⟩ 77483

def event77485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28283⟩⟩) 1 ⟨28282⟩ 77305

def event77486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28283⟩⟩) (.sum [.predecessor 0 77484 .coefficient, .predecessor 1 77485 .coefficient])

def event77487 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28283⟩⟩, .operator (⟨77483, 0⟩, ⟨77305, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28280⟩⟩]⟩, (1)⟩)

def event77488 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28283⟩⟩, .operator (⟨77483, 2⟩, ⟨77305, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24284⟩⟩]⟩, (-1)⟩)

def event77489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28283⟩⟩) (.sum [.result 77483 .summary, .result 77305 .summary])

def exact77490RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77490RawTermsValid :
    exact77490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77490 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28283⟩⟩) exact77490RawTerms .large 77486 (.finite 1292180536164689260544) (some (77489))

def event77491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28284⟩⟩) 0 ⟨28283⟩ 77490

def event77492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28284⟩⟩) 1 ⟨6682⟩ 5679

def event77493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28284⟩⟩) (.product (.predecessor 0 77491 .coefficient) (.predecessor 1 77492 .coefficient) (⟨false, false, none, none, none⟩))

def event77494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28284⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) [⟨.result 5675 .coefficient, false, none⟩])

def event77495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28284⟩⟩) (.product (.result 77490 .summary) (.transfer 77494) (⟨false, false, none, none, none⟩))

def event77496 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28284⟩⟩, .operator (⟨77490, 0⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩)

def event77497 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28284⟩⟩, .operator (⟨77490, 1⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (-1)⟩)

def event77498 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28284⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6681⟩⟩) ⟨6612⟩ 5672)

def event77499 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28284⟩⟩, .relation 77498 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77500RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77500RawTermsValid :
    exact77500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28284⟩⟩) exact77500RawTerms .large 77493 (.finite 4742323242612988221224648704) (some (77495))

def event77501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24221⟩⟩) 0 ⟨6689⟩ 5477

def event77502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24221⟩⟩) 1 ⟨24220⟩ 69627

def event77503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24221⟩⟩) (.authority (.operator))

def exact77504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩]

theorem exact77504RawTermsValid :
    exact77504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24221⟩⟩) exact77504RawTerms .large 77503 .exactZero (none)

def event77505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28063⟩⟩) 0 ⟨24221⟩ 77504

def event77506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28063⟩⟩) (.authority (.operator))

def exact77507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩]

theorem exact77507RawTermsValid :
    exact77507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77507 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28063⟩⟩) exact77507RawTerms (.finite 8192) 77506 .exactZero (none)

def event77508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28065⟩⟩) 0 ⟨26140⟩ 69911

def event77509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28065⟩⟩) 1 ⟨28063⟩ 77507

def event77510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28065⟩⟩) (.product (.predecessor 0 77508 .coefficient) (.predecessor 1 77509 .coefficient) (⟨false, false, none, none, none⟩))

def event77511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28065⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩) [⟨.result 77507 .coefficient, false, none⟩])

def event77512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28065⟩⟩) (.product (.result 69911 .summary) (.transfer 77511) (⟨false, false, none, none, none⟩))

def event77513 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28065⟩⟩, .operator (⟨69911, 0⟩, ⟨77507, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩)

def event77514 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28065⟩⟩, .operator (⟨69911, 1⟩, ⟨77507, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩)

def event77515 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28065⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28063⟩⟩) ⟨24221⟩ 77504)

def event77516 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28065⟩⟩, .relation 77515 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (-1)⟩)

def exact77517RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (-1)⟩]

theorem exact77517RawTermsValid :
    exact77517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28065⟩⟩) exact77517RawTerms .large 77510 (.finite 1292113297018323992576) (some (77512))

def event77518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21468⟩⟩) 0 ⟨16056⟩ 3310

def event77519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21468⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact77520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩]

theorem exact77520RawTermsValid :
    exact77520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77520 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21468⟩⟩) exact77520RawTerms (.finite 136065468) 77519 .exactZero (none)

def event77521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21470⟩⟩) 0 ⟨21468⟩ 77520

def event77522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21470⟩⟩) 1 ⟨2348⟩ 4

def event77523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21470⟩⟩) (.scale (.predecessor 0 77521 .coefficient) (.value (.predecessor 1 77522 .coefficient)))

def exact77524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩]

theorem exact77524RawTermsValid :
    exact77524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77524 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21470⟩⟩) exact77524RawTerms (.finite 136065468) 77523 .exactZero (none)

def event77525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21471⟩⟩) 0 ⟨5535⟩ 65387

def event77526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21471⟩⟩) 1 ⟨21470⟩ 77524

def event77527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21471⟩⟩) (.product (.predecessor 0 77525 .coefficient) (.predecessor 1 77526 .coefficient) (⟨false, false, none, none, none⟩))

def event77528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21471⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩) [⟨.result 77520 .coefficient, false, none⟩])

def event77529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21471⟩⟩) (.product (.result 65387 .summary) (.transfer 77528) (⟨false, false, none, none, none⟩))

def event77530 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21471⟩⟩, .operator (⟨65387, 0⟩, ⟨77524, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩)

def event77531 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21469⟩⟩)

def event77532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77533 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77535 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77537 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77539 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77539

def event77541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77537

def event77542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77540 .coefficient) (.value (.predecessor 1 77541 .coefficient)))

def event77543 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77543

def event77545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77535

def event77546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77544 .coefficient, .predecessor 1 77545 .coefficient])

def event77547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77547

def event77549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77533

def event77550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77549 .coefficient))

def event77551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 77551

def event77553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact77554RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact77554RawTermsValid :
    exact77554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77554 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact77554RawTerms (.finite 22) 77553 .exactZero (none)

def event77555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 77551

def event77556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact77557RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact77557RawTermsValid :
    exact77557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact77557RawTerms (.finite 22) 77556 .exactZero (none)

def event77558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 77557

def event77559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 77554

def event77560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 77558 .coefficient) (.predecessor 1 77559 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩) [⟨.result 77557 .coefficient, true, some 1⟩, ⟨.result 77554 .coefficient, true, some 1⟩])

def event77562 : Event := .survivorFold (1) 77561

def exact77563RawTerms : List Term := []

theorem exact77563RawTermsValid :
    exact77563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77563 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact77563RawTerms (.finite 484) 77560 (.finite 484) (some (77561))

def event77564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 77563

def event77565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 77564 .coefficient))

def event77566 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event77567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16055⟩⟩) 0 ⟨14417⟩ 77566

def eventLeaf4832 : Array AnnotatedEvent := #[
  { event := event77312
    frameStart := 0 },
  { event := event77313
    frameStart := 0 },
  { event := event77314
    frameStart := 0 },
  { event := event77315
    frameStart := 0 },
  { event := event77316
    frameStart := 0 },
  { event := event77317
    frameStart := 0 },
  { event := event77318
    frameStart := 0 },
  { event := event77319
    frameStart := 77319 },
  { event := event77320
    frameStart := 77319 },
  { event := event77321
    frameStart := 77319 },
  { event := event77322
    frameStart := 77319 },
  { event := event77323
    frameStart := 77319 },
  { event := event77324
    frameStart := 77319 },
  { event := event77325
    frameStart := 77319 },
  { event := event77326
    frameStart := 77319 },
  { event := event77327
    frameStart := 77319 }
]

def eventLeaf4833 : Array AnnotatedEvent := #[
  { event := event77328
    frameStart := 77319 },
  { event := event77329
    frameStart := 77319 },
  { event := event77330
    frameStart := 77319 },
  { event := event77331
    frameStart := 77319 },
  { event := event77332
    frameStart := 77319 },
  { event := event77333
    frameStart := 77319 },
  { event := event77334
    frameStart := 77319 },
  { event := event77335
    frameStart := 77319 },
  { event := event77336
    frameStart := 77319 },
  { event := event77337
    frameStart := 77319 },
  { event := event77338
    frameStart := 77319 },
  { event := event77339
    frameStart := 77319 },
  { event := event77340
    frameStart := 77319 },
  { event := event77341
    frameStart := 77319 },
  { event := event77342
    frameStart := 77319 },
  { event := event77343
    frameStart := 77319 }
]

def eventLeaf4834 : Array AnnotatedEvent := #[
  { event := event77344
    frameStart := 77319 },
  { event := event77345
    frameStart := 77319 },
  { event := event77346
    frameStart := 77319 },
  { event := event77347
    frameStart := 77319 },
  { event := event77348
    frameStart := 77319 },
  { event := event77349
    frameStart := 77319 },
  { event := event77350
    frameStart := 77319 },
  { event := event77351
    frameStart := 77319 },
  { event := event77352
    frameStart := 77319 },
  { event := event77353
    frameStart := 77319 },
  { event := event77354
    frameStart := 77319 },
  { event := event77355
    frameStart := 77319 },
  { event := event77356
    frameStart := 77319 },
  { event := event77357
    frameStart := 77319 },
  { event := event77358
    frameStart := 77319 },
  { event := event77359
    frameStart := 77319 }
]

def eventLeaf4835 : Array AnnotatedEvent := #[
  { event := event77360
    frameStart := 77319 },
  { event := event77361
    frameStart := 77319 },
  { event := event77362
    frameStart := 77319 },
  { event := event77363
    frameStart := 77319 },
  { event := event77364
    frameStart := 77319 },
  { event := event77365
    frameStart := 77319 },
  { event := event77366
    frameStart := 77319 },
  { event := event77367
    frameStart := 77319 },
  { event := event77368
    frameStart := 77319 },
  { event := event77369
    frameStart := 77319 },
  { event := event77370
    frameStart := 77319 },
  { event := event77371
    frameStart := 77319 },
  { event := event77372
    frameStart := 77319 },
  { event := event77373
    frameStart := 77373 },
  { event := event77374
    frameStart := 77373 },
  { event := event77375
    frameStart := 77373 }
]

def eventLeaf4836 : Array AnnotatedEvent := #[
  { event := event77376
    frameStart := 77373 },
  { event := event77377
    frameStart := 77373 },
  { event := event77378
    frameStart := 77373 },
  { event := event77379
    frameStart := 77373 },
  { event := event77380
    frameStart := 77373 },
  { event := event77381
    frameStart := 77373 },
  { event := event77382
    frameStart := 77373 },
  { event := event77383
    frameStart := 77373 },
  { event := event77384
    frameStart := 77373 },
  { event := event77385
    frameStart := 77373 },
  { event := event77386
    frameStart := 77373 },
  { event := event77387
    frameStart := 77373 },
  { event := event77388
    frameStart := 77373 },
  { event := event77389
    frameStart := 77373 },
  { event := event77390
    frameStart := 77373 },
  { event := event77391
    frameStart := 77373 }
]

def eventLeaf4837 : Array AnnotatedEvent := #[
  { event := event77392
    frameStart := 77373 },
  { event := event77393
    frameStart := 77373 },
  { event := event77394
    frameStart := 77373 },
  { event := event77395
    frameStart := 77373 },
  { event := event77396
    frameStart := 77373 },
  { event := event77397
    frameStart := 77373 },
  { event := event77398
    frameStart := 77373 },
  { event := event77399
    frameStart := 77373 },
  { event := event77400
    frameStart := 77373 },
  { event := event77401
    frameStart := 77373 },
  { event := event77402
    frameStart := 77373 },
  { event := event77403
    frameStart := 77373 },
  { event := event77404
    frameStart := 77373 },
  { event := event77405
    frameStart := 77373 },
  { event := event77406
    frameStart := 77373 },
  { event := event77407
    frameStart := 77373 }
]

def eventLeaf4838 : Array AnnotatedEvent := #[
  { event := event77408
    frameStart := 77373 },
  { event := event77409
    frameStart := 77373 },
  { event := event77410
    frameStart := 77373 },
  { event := event77411
    frameStart := 77373 },
  { event := event77412
    frameStart := 77373 },
  { event := event77413
    frameStart := 77373 },
  { event := event77414
    frameStart := 77373 },
  { event := event77415
    frameStart := 77373 },
  { event := event77416
    frameStart := 77373 },
  { event := event77417
    frameStart := 77373 },
  { event := event77418
    frameStart := 77373 },
  { event := event77419
    frameStart := 77373 },
  { event := event77420
    frameStart := 77373 },
  { event := event77421
    frameStart := 77373 },
  { event := event77422
    frameStart := 77373 },
  { event := event77423
    frameStart := 77373 }
]

def eventLeaf4839 : Array AnnotatedEvent := #[
  { event := event77424
    frameStart := 77373 },
  { event := event77425
    frameStart := 77373 },
  { event := event77426
    frameStart := 77373 },
  { event := event77427
    frameStart := 77373 },
  { event := event77428
    frameStart := 77373 },
  { event := event77429
    frameStart := 77373 },
  { event := event77430
    frameStart := 77373 },
  { event := event77431
    frameStart := 77373 },
  { event := event77432
    frameStart := 77373 },
  { event := event77433
    frameStart := 77373 },
  { event := event77434
    frameStart := 77373 },
  { event := event77435
    frameStart := 77373 },
  { event := event77436
    frameStart := 77373 },
  { event := event77437
    frameStart := 77373 },
  { event := event77438
    frameStart := 77373 },
  { event := event77439
    frameStart := 77373 }
]

def eventLeaf4840 : Array AnnotatedEvent := #[
  { event := event77440
    frameStart := 77373 },
  { event := event77441
    frameStart := 77373 },
  { event := event77442
    frameStart := 77373 },
  { event := event77443
    frameStart := 77373 },
  { event := event77444
    frameStart := 77373 },
  { event := event77445
    frameStart := 77373 },
  { event := event77446
    frameStart := 77373 },
  { event := event77447
    frameStart := 77373 },
  { event := event77448
    frameStart := 77373 },
  { event := event77449
    frameStart := 77373 },
  { event := event77450
    frameStart := 77373 },
  { event := event77451
    frameStart := 77373 },
  { event := event77452
    frameStart := 77373 },
  { event := event77453
    frameStart := 77373 },
  { event := event77454
    frameStart := 77373 },
  { event := event77455
    frameStart := 77373 }
]

def eventLeaf4841 : Array AnnotatedEvent := #[
  { event := event77456
    frameStart := 77373 },
  { event := event77457
    frameStart := 77373 },
  { event := event77458
    frameStart := 77373 },
  { event := event77459
    frameStart := 77373 },
  { event := event77460
    frameStart := 77373 },
  { event := event77461
    frameStart := 77373 },
  { event := event77462
    frameStart := 77373 },
  { event := event77463
    frameStart := 77373 },
  { event := event77464
    frameStart := 77373 },
  { event := event77465
    frameStart := 77373 },
  { event := event77466
    frameStart := 77373 },
  { event := event77467
    frameStart := 77373 },
  { event := event77468
    frameStart := 77373 },
  { event := event77469
    frameStart := 77373 },
  { event := event77470
    frameStart := 77373 },
  { event := event77471
    frameStart := 77373 }
]

def eventLeaf4842 : Array AnnotatedEvent := #[
  { event := event77472
    frameStart := 77373 },
  { event := event77473
    frameStart := 77373 },
  { event := event77474
    frameStart := 77373 },
  { event := event77475
    frameStart := 77373 },
  { event := event77476
    frameStart := 77373 },
  { event := event77477
    frameStart := 0 },
  { event := event77478
    frameStart := 0 },
  { event := event77479
    frameStart := 0 },
  { event := event77480
    frameStart := 0 },
  { event := event77481
    frameStart := 0 },
  { event := event77482
    frameStart := 0 },
  { event := event77483
    frameStart := 0 },
  { event := event77484
    frameStart := 0 },
  { event := event77485
    frameStart := 0 },
  { event := event77486
    frameStart := 0 },
  { event := event77487
    frameStart := 0 }
]

def eventLeaf4843 : Array AnnotatedEvent := #[
  { event := event77488
    frameStart := 0 },
  { event := event77489
    frameStart := 0 },
  { event := event77490
    frameStart := 0 },
  { event := event77491
    frameStart := 0 },
  { event := event77492
    frameStart := 0 },
  { event := event77493
    frameStart := 0 },
  { event := event77494
    frameStart := 0 },
  { event := event77495
    frameStart := 0 },
  { event := event77496
    frameStart := 0 },
  { event := event77497
    frameStart := 0 },
  { event := event77498
    frameStart := 0 },
  { event := event77499
    frameStart := 0 },
  { event := event77500
    frameStart := 0 },
  { event := event77501
    frameStart := 0 },
  { event := event77502
    frameStart := 0 },
  { event := event77503
    frameStart := 0 }
]

def eventLeaf4844 : Array AnnotatedEvent := #[
  { event := event77504
    frameStart := 0 },
  { event := event77505
    frameStart := 0 },
  { event := event77506
    frameStart := 0 },
  { event := event77507
    frameStart := 0 },
  { event := event77508
    frameStart := 0 },
  { event := event77509
    frameStart := 0 },
  { event := event77510
    frameStart := 0 },
  { event := event77511
    frameStart := 0 },
  { event := event77512
    frameStart := 0 },
  { event := event77513
    frameStart := 0 },
  { event := event77514
    frameStart := 0 },
  { event := event77515
    frameStart := 0 },
  { event := event77516
    frameStart := 0 },
  { event := event77517
    frameStart := 0 },
  { event := event77518
    frameStart := 0 },
  { event := event77519
    frameStart := 0 }
]

def eventLeaf4845 : Array AnnotatedEvent := #[
  { event := event77520
    frameStart := 0 },
  { event := event77521
    frameStart := 0 },
  { event := event77522
    frameStart := 0 },
  { event := event77523
    frameStart := 0 },
  { event := event77524
    frameStart := 0 },
  { event := event77525
    frameStart := 0 },
  { event := event77526
    frameStart := 0 },
  { event := event77527
    frameStart := 0 },
  { event := event77528
    frameStart := 0 },
  { event := event77529
    frameStart := 0 },
  { event := event77530
    frameStart := 0 },
  { event := event77531
    frameStart := 77531 },
  { event := event77532
    frameStart := 77531 },
  { event := event77533
    frameStart := 77531 },
  { event := event77534
    frameStart := 77531 },
  { event := event77535
    frameStart := 77531 }
]

def eventLeaf4846 : Array AnnotatedEvent := #[
  { event := event77536
    frameStart := 77531 },
  { event := event77537
    frameStart := 77531 },
  { event := event77538
    frameStart := 77531 },
  { event := event77539
    frameStart := 77531 },
  { event := event77540
    frameStart := 77531 },
  { event := event77541
    frameStart := 77531 },
  { event := event77542
    frameStart := 77531 },
  { event := event77543
    frameStart := 77531 },
  { event := event77544
    frameStart := 77531 },
  { event := event77545
    frameStart := 77531 },
  { event := event77546
    frameStart := 77531 },
  { event := event77547
    frameStart := 77531 },
  { event := event77548
    frameStart := 77531 },
  { event := event77549
    frameStart := 77531 },
  { event := event77550
    frameStart := 77531 },
  { event := event77551
    frameStart := 77531 }
]

def eventLeaf4847 : Array AnnotatedEvent := #[
  { event := event77552
    frameStart := 77531 },
  { event := event77553
    frameStart := 77531 },
  { event := event77554
    frameStart := 77531 },
  { event := event77555
    frameStart := 77531 },
  { event := event77556
    frameStart := 77531 },
  { event := event77557
    frameStart := 77531 },
  { event := event77558
    frameStart := 77531 },
  { event := event77559
    frameStart := 77531 },
  { event := event77560
    frameStart := 77531 },
  { event := event77561
    frameStart := 77531 },
  { event := event77562
    frameStart := 77531 },
  { event := event77563
    frameStart := 77531 },
  { event := event77564
    frameStart := 77531 },
  { event := event77565
    frameStart := 77531 },
  { event := event77566
    frameStart := 77531 },
  { event := event77567
    frameStart := 77531 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events302
