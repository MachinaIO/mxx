import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events216

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact55296RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (-1)⟩]

theorem exact55296RawTermsValid :
    exact55296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55296 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28098⟩⟩) exact55296RawTerms .large 55289 (.finite 1292113297018323992576) (some (55291))

def event55297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21548⟩⟩) 0 ⟨16064⟩ 2562

def event55298 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21548⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact55299RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩]

theorem exact55299RawTermsValid :
    exact55299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55299 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21548⟩⟩) exact55299RawTerms (.finite 136065468) 55298 .exactZero (none)

def event55300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21550⟩⟩) 0 ⟨21548⟩ 55299

def event55301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21550⟩⟩) 1 ⟨2348⟩ 4

def event55302 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21550⟩⟩) (.scale (.predecessor 0 55300 .coefficient) (.value (.predecessor 1 55301 .coefficient)))

def exact55303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩]

theorem exact55303RawTermsValid :
    exact55303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21550⟩⟩) exact55303RawTerms (.finite 136065468) 55302 .exactZero (none)

def event55304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21551⟩⟩) 0 ⟨5547⟩ 50762

def event55305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21551⟩⟩) 1 ⟨21550⟩ 55303

def event55306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21551⟩⟩) (.product (.predecessor 0 55304 .coefficient) (.predecessor 1 55305 .coefficient) (⟨false, false, none, none, none⟩))

def event55307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21551⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) [⟨.result 55299 .coefficient, false, none⟩])

def event55308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21551⟩⟩) (.product (.result 50762 .summary) (.transfer 55307) (⟨false, false, none, none, none⟩))

def event55309 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21551⟩⟩, .operator (⟨50762, 0⟩, ⟨55303, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩)

def event55310 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21549⟩⟩)

def event55311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event55312 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event55313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event55314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event55315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event55316 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event55317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event55318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event55319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 55318

def event55320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 55316

def event55321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 55319 .coefficient) (.value (.predecessor 1 55320 .coefficient)))

def event55322 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event55323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 55322

def event55324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 55314

def event55325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 55323 .coefficient, .predecessor 1 55324 .coefficient])

def event55326 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event55327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 55326

def event55328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 55312

def event55329 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 55328 .coefficient))

def event55330 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event55331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11557⟩⟩) 0 ⟨5542⟩ 55330

def event55332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11557⟩⟩) (.authority (.programFamilyFact))

def exact55333RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩], []⟩, (1)⟩]

theorem exact55333RawTermsValid :
    exact55333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11557⟩⟩) exact55333RawTerms (.finite 22) 55332 .exactZero (none)

def event55334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14433⟩⟩) 0 ⟨5542⟩ 55330

def event55335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14433⟩⟩) (.authority (.programFamilyFact))

def exact55336RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩, (1)⟩]

theorem exact55336RawTermsValid :
    exact55336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14433⟩⟩) exact55336RawTerms (.finite 22) 55335 .exactZero (none)

def event55337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14434⟩⟩) 0 ⟨14433⟩ 55336

def event55338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14434⟩⟩) 1 ⟨11557⟩ 55333

def event55339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14434⟩⟩) (.product (.predecessor 0 55337 .coefficient) (.predecessor 1 55338 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14434⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩) [⟨.result 55336 .coefficient, true, some 1⟩, ⟨.result 55333 .coefficient, true, some 1⟩])

def event55341 : Event := .survivorFold (1) 55340

def exact55342RawTerms : List Term := []

theorem exact55342RawTermsValid :
    exact55342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55342 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14434⟩⟩) exact55342RawTerms (.finite 484) 55339 (.finite 484) (some (55340))

def event55343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14435⟩⟩) 0 ⟨14434⟩ 55342

def event55344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14435⟩⟩) (.identity (.predecessor 0 55343 .coefficient))

def event55345 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14435⟩⟩) (.finite 484)

def event55346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16063⟩⟩) 0 ⟨14435⟩ 55345

def event55347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16063⟩⟩) (.authority (.programFamilyFact))

def exact55348RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], []⟩, (1)⟩]

theorem exact55348RawTermsValid :
    exact55348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16063⟩⟩) exact55348RawTerms (.finite 22) 55347 .exactZero (none)

def event55349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16064⟩⟩) 0 ⟨16063⟩ 55348

def event55350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16064⟩⟩) (.identity (.predecessor 0 55349 .coefficient))

def event55351 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16064⟩⟩) (.finite 22)

def event55352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21548⟩⟩) 0 ⟨16064⟩ 55351

def event55353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21548⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact55354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩]

theorem exact55354RawTermsValid :
    exact55354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55354 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21548⟩⟩) exact55354RawTerms (.finite 136065468) 55353 .exactZero (none)

def event55355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact55356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact55356RawTermsValid :
    exact55356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55356 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact55356RawTerms .large 55355 .exactZero (none)

def event55357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21549⟩⟩) 0 ⟨6⟩ 55356

def event55358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21549⟩⟩) 1 ⟨21548⟩ 55354

def event55359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21549⟩⟩) (.product (.predecessor 0 55357 .coefficient) (.predecessor 1 55358 .coefficient) (⟨false, false, none, none, none⟩))

def event55360 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21549⟩⟩, .operator (⟨55356, 0⟩, ⟨55354, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩)

def exact55361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩]

theorem exact55361RawTermsValid :
    exact55361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21549⟩⟩) exact55361RawTerms .large 55359 .exactZero (none)

def event55362 : Event := .preFoldPolynomial 55361 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩] .exactZero none

def exact55363RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩, (1)⟩]

def event55363 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21549⟩⟩) 55362 exact55363RawTerms .large 55359 .exactZero (none)

def event55364 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28101⟩⟩)

def event55365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event55366 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event55367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event55368 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event55369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event55370 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event55371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event55372 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event55373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 55372

def event55374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 55370

def event55375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 55373 .coefficient) (.value (.predecessor 1 55374 .coefficient)))

def event55376 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event55377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 55376

def event55378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 55368

def event55379 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 55377 .coefficient, .predecessor 1 55378 .coefficient])

def event55380 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event55381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 55380

def event55382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 55366

def event55383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 55382 .coefficient))

def event55384 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event55385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11557⟩⟩) 0 ⟨5542⟩ 55384

def event55386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11557⟩⟩) (.authority (.programFamilyFact))

def exact55387RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩], []⟩, (1)⟩]

theorem exact55387RawTermsValid :
    exact55387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55387 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11557⟩⟩) exact55387RawTerms (.finite 22) 55386 .exactZero (none)

def event55388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14433⟩⟩) 0 ⟨5542⟩ 55384

def event55389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14433⟩⟩) (.authority (.programFamilyFact))

def exact55390RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩, (1)⟩]

theorem exact55390RawTermsValid :
    exact55390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55390 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14433⟩⟩) exact55390RawTerms (.finite 22) 55389 .exactZero (none)

def event55391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14434⟩⟩) 0 ⟨14433⟩ 55390

def event55392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14434⟩⟩) 1 ⟨11557⟩ 55387

def event55393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14434⟩⟩) (.product (.predecessor 0 55391 .coefficient) (.predecessor 1 55392 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55394 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14434⟩⟩, .operator (⟨55390, 0⟩, ⟨55387, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩, (1)⟩)

def exact55395RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11557⟩⟩, ⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩, (1)⟩]

theorem exact55395RawTermsValid :
    exact55395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14434⟩⟩) exact55395RawTerms (.finite 484) 55393 .exactZero (none)

def event55396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14435⟩⟩) 0 ⟨14434⟩ 55395

def event55397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14435⟩⟩) (.identity (.predecessor 0 55396 .coefficient))

def event55398 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14435⟩⟩) (.finite 484)

def event55399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16063⟩⟩) 0 ⟨14435⟩ 55398

def event55400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16063⟩⟩) (.authority (.programFamilyFact))

def exact55401RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], []⟩, (1)⟩]

theorem exact55401RawTermsValid :
    exact55401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55401 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16063⟩⟩) exact55401RawTerms (.finite 22) 55400 .exactZero (none)

def event55402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16064⟩⟩) 0 ⟨16063⟩ 55401

def event55403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16064⟩⟩) (.identity (.predecessor 0 55402 .coefficient))

def event55404 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16064⟩⟩) (.finite 22)

def event55405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24226⟩⟩) 0 ⟨16064⟩ 55404

def event55406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24226⟩⟩) (.authority (.programFamilyFact))

def event55407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24226⟩⟩) (.finite 3720)

def event55408 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event55409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24228⟩⟩) 0 ⟨6689⟩ 55408

def event55410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24228⟩⟩) 1 ⟨24226⟩ 55407

def event55411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24228⟩⟩) (.authority (.operator))

def exact55412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩]

theorem exact55412RawTermsValid :
    exact55412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55412 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24228⟩⟩) exact55412RawTerms .large 55411 .exactZero (none)

def event55413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28096⟩⟩) 0 ⟨24228⟩ 55412

def event55414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28096⟩⟩) (.authority (.operator))

def exact55415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩]

theorem exact55415RawTermsValid :
    exact55415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55415 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28096⟩⟩) exact55415RawTerms (.finite 8192) 55414 .exactZero (none)

def event55416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event55417 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event55418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16138⟩⟩) 0 ⟨16064⟩ 55404

def event55419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16138⟩⟩) 1 ⟨110⟩ 55417

def event55420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16138⟩⟩) (.sum [.predecessor 0 55418 .coefficient, .predecessor 1 55419 .coefficient])

def event55421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16138⟩⟩) (.finite 22)

def event55422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16139⟩⟩) 0 ⟨16138⟩ 55421

def event55423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16139⟩⟩) (.identity (.predecessor 0 55422 .coefficient))

def exact55424RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], []⟩, (1)⟩]

theorem exact55424RawTermsValid :
    exact55424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16139⟩⟩) exact55424RawTerms (.finite 22) 55423 .exactZero (none)

def event55425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact55426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55426RawTermsValid :
    exact55426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact55426RawTerms .large 55425 .exactZero (none)

def event55427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16140⟩⟩) 0 ⟨6544⟩ 55426

def event55428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16140⟩⟩) 1 ⟨16139⟩ 55424

def event55429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16140⟩⟩) (.product (.predecessor 0 55427 .coefficient) (.predecessor 1 55428 .coefficient) (⟨false, false, none, none, none⟩))

def event55430 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16140⟩⟩, .operator (⟨55426, 0⟩, ⟨55424, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55431RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55431RawTermsValid :
    exact55431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16140⟩⟩) exact55431RawTerms .large 55429 .exactZero (none)

def event55432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 55408

def event55433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact55434RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact55434RawTermsValid :
    exact55434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55434 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact55434RawTerms .large 55433 .exactZero (none)

def event55435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16141⟩⟩) 0 ⟨6698⟩ 55434

def event55436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16141⟩⟩) 1 ⟨16140⟩ 55431

def event55437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16141⟩⟩) (.sum [.predecessor 0 55435 .coefficient, .predecessor 1 55436 .coefficient])

def exact55438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55438RawTermsValid :
    exact55438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55438 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16141⟩⟩) exact55438RawTerms .large 55437 .exactZero (none)

def event55439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28097⟩⟩) 0 ⟨16141⟩ 55438

def event55440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28097⟩⟩) 1 ⟨28096⟩ 55415

def event55441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28097⟩⟩) (.product (.predecessor 0 55439 .coefficient) (.predecessor 1 55440 .coefficient) (⟨false, false, none, none, none⟩))

def event55442 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28097⟩⟩, .operator (⟨55438, 0⟩, ⟨55415, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩)

def event55443 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28097⟩⟩, .operator (⟨55438, 1⟩, ⟨55415, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩)

def event55444 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28097⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28096⟩⟩) ⟨24228⟩ 55412)

def event55445 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28097⟩⟩, .relation 55444 0, ⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (-1)⟩)

def exact55446RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (-1)⟩]

theorem exact55446RawTermsValid :
    exact55446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28097⟩⟩) exact55446RawTerms .large 55441 .exactZero (none)

def event55447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16108⟩⟩) 0 ⟨16064⟩ 55404

def event55448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16108⟩⟩) (.authority (.programFamilyFact))

def exact55449RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩]

theorem exact55449RawTermsValid :
    exact55449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55449 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16108⟩⟩) exact55449RawTerms (.finite 61) 55448 .exactZero (none)

def event55450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16109⟩⟩) 0 ⟨6544⟩ 55426

def event55451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16109⟩⟩) 1 ⟨16108⟩ 55449

def event55452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16109⟩⟩) (.product (.predecessor 0 55450 .coefficient) (.predecessor 1 55451 .coefficient) (⟨false, true, none, none, some 1⟩))

def event55453 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16109⟩⟩, .operator (⟨55426, 0⟩, ⟨55449, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55454RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55454RawTermsValid :
    exact55454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55454 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16109⟩⟩) exact55454RawTerms .large 55452 .exactZero (none)

def event55455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6725⟩⟩) 0 ⟨6689⟩ 55408

def event55456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6725⟩⟩) (.authority (.operator))

def exact55457RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩]

theorem exact55457RawTermsValid :
    exact55457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6725⟩⟩) exact55457RawTerms .large 55456 .exactZero (none)

def event55458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16110⟩⟩) 0 ⟨6725⟩ 55457

def event55459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16110⟩⟩) 1 ⟨16109⟩ 55454

def event55460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16110⟩⟩) (.sum [.predecessor 0 55458 .coefficient, .predecessor 1 55459 .coefficient])

def exact55461RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55461RawTermsValid :
    exact55461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16110⟩⟩) exact55461RawTerms .large 55460 .exactZero (none)

def event55462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28101⟩⟩) 0 ⟨16110⟩ 55461

def event55463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28101⟩⟩) 1 ⟨28097⟩ 55446

def event55464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28101⟩⟩) (.sum [.predecessor 0 55462 .coefficient, .predecessor 1 55463 .coefficient])

def exact55465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55465RawTermsValid :
    exact55465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55465 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28101⟩⟩) exact55465RawTerms .large 55464 .exactZero (none)

def event55466 : Event := .preFoldPolynomial 55465 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact55467RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event55467 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28101⟩⟩) 55466 exact55467RawTerms .large 55464 .exactZero (none)

def event55468 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16064⟩⟩) ⟨⟨138⟩, ⟨46⟩, ⟨109⟩⟩ ⟨55310, 55468⟩

def event55469 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21551⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (1) 0 2 (.universal 55468 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21548⟩⟩]⟩) (none) 55467)

def event55470 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21551⟩⟩, .relation 55469 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩)

def event55471 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21551⟩⟩, .relation 55469 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩)

def event55472 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21551⟩⟩, .relation 55469 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩)

def event55473 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21551⟩⟩, .relation 55469 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact55474RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55474RawTermsValid :
    exact55474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21551⟩⟩) exact55474RawTerms .large 55306 (.finite 1811303510016) (some (55308))

def event55475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28099⟩⟩) 0 ⟨21551⟩ 55474

def event55476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28099⟩⟩) 1 ⟨28098⟩ 55296

def event55477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28099⟩⟩) (.sum [.predecessor 0 55475 .coefficient, .predecessor 1 55476 .coefficient])

def event55478 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28099⟩⟩, .operator (⟨55474, 0⟩, ⟨55296, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩)

def event55479 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28099⟩⟩, .operator (⟨55474, 2⟩, ⟨55296, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16063⟩⟩], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (-1)⟩)

def event55480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28099⟩⟩) (.sum [.result 55474 .summary, .result 55296 .summary])

def exact55481RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55481RawTermsValid :
    exact55481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28099⟩⟩) exact55481RawTerms .large 55477 (.finite 1292113298829627502592) (some (55480))

def event55482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24163⟩⟩) 0 ⟨15945⟩ 2585

def event55483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24163⟩⟩) (.authority (.programFamilyFact))

def event55484 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24163⟩⟩) (.finite 3720)

def event55485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24165⟩⟩) 0 ⟨6689⟩ 5477

def event55486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24165⟩⟩) 1 ⟨24163⟩ 55484

def event55487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24165⟩⟩) (.authority (.operator))

def exact55488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24165⟩⟩]⟩, (1)⟩]

theorem exact55488RawTermsValid :
    exact55488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24165⟩⟩) exact55488RawTerms .large 55487 .exactZero (none)

def event55489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27879⟩⟩) 0 ⟨24165⟩ 55488

def event55490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27879⟩⟩) (.authority (.operator))

def exact55491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩, (1)⟩]

theorem exact55491RawTermsValid :
    exact55491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27879⟩⟩) exact55491RawTerms (.finite 8192) 55490 .exactZero (none)

def event55492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23585⟩⟩) 0 ⟨14218⟩ 2579

def event55493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23585⟩⟩) (.authority (.programFamilyFact))

def event55494 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23585⟩⟩) (.finite 3720)

def event55495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23586⟩⟩) 0 ⟨6689⟩ 5477

def event55496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23586⟩⟩) 1 ⟨23585⟩ 55494

def event55497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23586⟩⟩) (.authority (.operator))

def exact55498RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩]

theorem exact55498RawTermsValid :
    exact55498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55498 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23586⟩⟩) exact55498RawTerms .large 55497 .exactZero (none)

def event55499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26071⟩⟩) 0 ⟨23586⟩ 55498

def event55500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26071⟩⟩) (.authority (.operator))

def exact55501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩]

theorem exact55501RawTermsValid :
    exact55501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55501 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26071⟩⟩) exact55501RawTerms (.finite 8192) 55500 .exactZero (none)

def event55502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11474⟩⟩) 0 ⟨11473⟩ 2568

def event55503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11474⟩⟩) 1 ⟨6568⟩ 50670

def event55504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11474⟩⟩) (.tensor (.predecessor 0 55502 .coefficient) (.predecessor 1 55503 .coefficient) true false)

def event55505 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11474⟩⟩, .operator (⟨2568, 0⟩, ⟨50670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55506RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55506RawTermsValid :
    exact55506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55506 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11474⟩⟩) exact55506RawTerms .large 55504 .exactZero (none)

def event55507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7273⟩⟩) 0 ⟨5545⟩ 50540

def event55508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7273⟩⟩) 1 ⟨6779⟩ 11482

def event55509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7273⟩⟩) (.product (.predecessor 0 55507 .coefficient) (.predecessor 1 55508 .coefficient) (⟨false, false, none, none, none⟩))

def event55510 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7273⟩⟩, .operator (⟨50540, 0⟩, ⟨11482, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact55511RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact55511RawTermsValid :
    exact55511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55511 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7273⟩⟩) exact55511RawTerms .large 55509 .exactZero (none)

def event55512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11475⟩⟩) 0 ⟨7273⟩ 55511

def event55513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11475⟩⟩) 1 ⟨11474⟩ 55506

def event55514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11475⟩⟩) (.sum [.predecessor 0 55512 .coefficient, .predecessor 1 55513 .coefficient])

def exact55515RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55515RawTermsValid :
    exact55515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55515 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11475⟩⟩) exact55515RawTerms .large 55514 .exactZero (none)

def event55516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11476⟩⟩) 0 ⟨11475⟩ 55515

def event55517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11476⟩⟩) 1 ⟨93⟩ 11474

def event55518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11476⟩⟩) (.sum [.predecessor 0 55516 .coefficient, .predecessor 1 55517 .coefficient])

def event55519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11476⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) [⟨.result 11474 .coefficient, false, none⟩])

def event55520 : Event := .survivorFold (1) 55519

def exact55521RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55521RawTermsValid :
    exact55521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11476⟩⟩) exact55521RawTerms .large 55518 (.finite 26) (some (55519))

def event55522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14219⟩⟩) 0 ⟨11476⟩ 55521

def event55523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14219⟩⟩) 1 ⟨14216⟩ 2571

def event55524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14219⟩⟩) (.product (.predecessor 0 55522 .coefficient) (.predecessor 1 55523 .coefficient) (⟨false, true, none, none, some 1⟩))

def event55525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14219⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩) [⟨.result 2571 .coefficient, true, some 1⟩])

def event55526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14219⟩⟩) (.product (.result 55521 .summary) (.transfer 55525) (⟨false, false, none, none, none⟩))

def event55527 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14219⟩⟩, .operator (⟨55521, 1⟩, ⟨2571, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event55528 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14219⟩⟩, .operator (⟨55521, 0⟩, ⟨2571, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact55529RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact55529RawTermsValid :
    exact55529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55529 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14219⟩⟩) exact55529RawTerms .large 55524 (.finite 14976) (some (55526))

def event55530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14220⟩⟩) 0 ⟨14216⟩ 2571

def event55531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14220⟩⟩) 1 ⟨6568⟩ 50670

def event55532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14220⟩⟩) (.tensor (.predecessor 0 55530 .coefficient) (.predecessor 1 55531 .coefficient) true false)

def event55533 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14220⟩⟩, .operator (⟨2571, 0⟩, ⟨50670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55534RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55534RawTermsValid :
    exact55534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14220⟩⟩) exact55534RawTerms .large 55532 .exactZero (none)

def event55535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7253⟩⟩) 0 ⟨5545⟩ 50540

def event55536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7253⟩⟩) 1 ⟨6759⟩ 11523

def event55537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7253⟩⟩) (.product (.predecessor 0 55535 .coefficient) (.predecessor 1 55536 .coefficient) (⟨false, false, none, none, none⟩))

def event55538 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7253⟩⟩, .operator (⟨50540, 0⟩, ⟨11523, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩)

def exact55539RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact55539RawTermsValid :
    exact55539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55539 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7253⟩⟩) exact55539RawTerms .large 55537 .exactZero (none)

def event55540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14221⟩⟩) 0 ⟨7253⟩ 55539

def event55541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14221⟩⟩) 1 ⟨14220⟩ 55534

def event55542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14221⟩⟩) (.sum [.predecessor 0 55540 .coefficient, .predecessor 1 55541 .coefficient])

def exact55543RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55543RawTermsValid :
    exact55543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55543 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14221⟩⟩) exact55543RawTerms .large 55542 .exactZero (none)

def event55544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14222⟩⟩) 0 ⟨14221⟩ 55543

def event55545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14222⟩⟩) 1 ⟨73⟩ 11515

def event55546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14222⟩⟩) (.sum [.predecessor 0 55544 .coefficient, .predecessor 1 55545 .coefficient])

def event55547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14222⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) [⟨.result 11515 .coefficient, false, none⟩])

def event55548 : Event := .survivorFold (1) 55547

def exact55549RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55549RawTermsValid :
    exact55549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14222⟩⟩) exact55549RawTerms .large 55546 (.finite 26) (some (55547))

def event55550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14223⟩⟩) 0 ⟨14222⟩ 55549

def event55551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14223⟩⟩) 1 ⟨7853⟩ 11512

def eventLeaf3456 : Array AnnotatedEvent := #[
  { event := event55296
    frameStart := 0 },
  { event := event55297
    frameStart := 0 },
  { event := event55298
    frameStart := 0 },
  { event := event55299
    frameStart := 0 },
  { event := event55300
    frameStart := 0 },
  { event := event55301
    frameStart := 0 },
  { event := event55302
    frameStart := 0 },
  { event := event55303
    frameStart := 0 },
  { event := event55304
    frameStart := 0 },
  { event := event55305
    frameStart := 0 },
  { event := event55306
    frameStart := 0 },
  { event := event55307
    frameStart := 0 },
  { event := event55308
    frameStart := 0 },
  { event := event55309
    frameStart := 0 },
  { event := event55310
    frameStart := 55310 },
  { event := event55311
    frameStart := 55310 }
]

def eventLeaf3457 : Array AnnotatedEvent := #[
  { event := event55312
    frameStart := 55310 },
  { event := event55313
    frameStart := 55310 },
  { event := event55314
    frameStart := 55310 },
  { event := event55315
    frameStart := 55310 },
  { event := event55316
    frameStart := 55310 },
  { event := event55317
    frameStart := 55310 },
  { event := event55318
    frameStart := 55310 },
  { event := event55319
    frameStart := 55310 },
  { event := event55320
    frameStart := 55310 },
  { event := event55321
    frameStart := 55310 },
  { event := event55322
    frameStart := 55310 },
  { event := event55323
    frameStart := 55310 },
  { event := event55324
    frameStart := 55310 },
  { event := event55325
    frameStart := 55310 },
  { event := event55326
    frameStart := 55310 },
  { event := event55327
    frameStart := 55310 }
]

def eventLeaf3458 : Array AnnotatedEvent := #[
  { event := event55328
    frameStart := 55310 },
  { event := event55329
    frameStart := 55310 },
  { event := event55330
    frameStart := 55310 },
  { event := event55331
    frameStart := 55310 },
  { event := event55332
    frameStart := 55310 },
  { event := event55333
    frameStart := 55310 },
  { event := event55334
    frameStart := 55310 },
  { event := event55335
    frameStart := 55310 },
  { event := event55336
    frameStart := 55310 },
  { event := event55337
    frameStart := 55310 },
  { event := event55338
    frameStart := 55310 },
  { event := event55339
    frameStart := 55310 },
  { event := event55340
    frameStart := 55310 },
  { event := event55341
    frameStart := 55310 },
  { event := event55342
    frameStart := 55310 },
  { event := event55343
    frameStart := 55310 }
]

def eventLeaf3459 : Array AnnotatedEvent := #[
  { event := event55344
    frameStart := 55310 },
  { event := event55345
    frameStart := 55310 },
  { event := event55346
    frameStart := 55310 },
  { event := event55347
    frameStart := 55310 },
  { event := event55348
    frameStart := 55310 },
  { event := event55349
    frameStart := 55310 },
  { event := event55350
    frameStart := 55310 },
  { event := event55351
    frameStart := 55310 },
  { event := event55352
    frameStart := 55310 },
  { event := event55353
    frameStart := 55310 },
  { event := event55354
    frameStart := 55310 },
  { event := event55355
    frameStart := 55310 },
  { event := event55356
    frameStart := 55310 },
  { event := event55357
    frameStart := 55310 },
  { event := event55358
    frameStart := 55310 },
  { event := event55359
    frameStart := 55310 }
]

def eventLeaf3460 : Array AnnotatedEvent := #[
  { event := event55360
    frameStart := 55310 },
  { event := event55361
    frameStart := 55310 },
  { event := event55362
    frameStart := 55310 },
  { event := event55363
    frameStart := 55310 },
  { event := event55364
    frameStart := 55364 },
  { event := event55365
    frameStart := 55364 },
  { event := event55366
    frameStart := 55364 },
  { event := event55367
    frameStart := 55364 },
  { event := event55368
    frameStart := 55364 },
  { event := event55369
    frameStart := 55364 },
  { event := event55370
    frameStart := 55364 },
  { event := event55371
    frameStart := 55364 },
  { event := event55372
    frameStart := 55364 },
  { event := event55373
    frameStart := 55364 },
  { event := event55374
    frameStart := 55364 },
  { event := event55375
    frameStart := 55364 }
]

def eventLeaf3461 : Array AnnotatedEvent := #[
  { event := event55376
    frameStart := 55364 },
  { event := event55377
    frameStart := 55364 },
  { event := event55378
    frameStart := 55364 },
  { event := event55379
    frameStart := 55364 },
  { event := event55380
    frameStart := 55364 },
  { event := event55381
    frameStart := 55364 },
  { event := event55382
    frameStart := 55364 },
  { event := event55383
    frameStart := 55364 },
  { event := event55384
    frameStart := 55364 },
  { event := event55385
    frameStart := 55364 },
  { event := event55386
    frameStart := 55364 },
  { event := event55387
    frameStart := 55364 },
  { event := event55388
    frameStart := 55364 },
  { event := event55389
    frameStart := 55364 },
  { event := event55390
    frameStart := 55364 },
  { event := event55391
    frameStart := 55364 }
]

def eventLeaf3462 : Array AnnotatedEvent := #[
  { event := event55392
    frameStart := 55364 },
  { event := event55393
    frameStart := 55364 },
  { event := event55394
    frameStart := 55364 },
  { event := event55395
    frameStart := 55364 },
  { event := event55396
    frameStart := 55364 },
  { event := event55397
    frameStart := 55364 },
  { event := event55398
    frameStart := 55364 },
  { event := event55399
    frameStart := 55364 },
  { event := event55400
    frameStart := 55364 },
  { event := event55401
    frameStart := 55364 },
  { event := event55402
    frameStart := 55364 },
  { event := event55403
    frameStart := 55364 },
  { event := event55404
    frameStart := 55364 },
  { event := event55405
    frameStart := 55364 },
  { event := event55406
    frameStart := 55364 },
  { event := event55407
    frameStart := 55364 }
]

def eventLeaf3463 : Array AnnotatedEvent := #[
  { event := event55408
    frameStart := 55364 },
  { event := event55409
    frameStart := 55364 },
  { event := event55410
    frameStart := 55364 },
  { event := event55411
    frameStart := 55364 },
  { event := event55412
    frameStart := 55364 },
  { event := event55413
    frameStart := 55364 },
  { event := event55414
    frameStart := 55364 },
  { event := event55415
    frameStart := 55364 },
  { event := event55416
    frameStart := 55364 },
  { event := event55417
    frameStart := 55364 },
  { event := event55418
    frameStart := 55364 },
  { event := event55419
    frameStart := 55364 },
  { event := event55420
    frameStart := 55364 },
  { event := event55421
    frameStart := 55364 },
  { event := event55422
    frameStart := 55364 },
  { event := event55423
    frameStart := 55364 }
]

def eventLeaf3464 : Array AnnotatedEvent := #[
  { event := event55424
    frameStart := 55364 },
  { event := event55425
    frameStart := 55364 },
  { event := event55426
    frameStart := 55364 },
  { event := event55427
    frameStart := 55364 },
  { event := event55428
    frameStart := 55364 },
  { event := event55429
    frameStart := 55364 },
  { event := event55430
    frameStart := 55364 },
  { event := event55431
    frameStart := 55364 },
  { event := event55432
    frameStart := 55364 },
  { event := event55433
    frameStart := 55364 },
  { event := event55434
    frameStart := 55364 },
  { event := event55435
    frameStart := 55364 },
  { event := event55436
    frameStart := 55364 },
  { event := event55437
    frameStart := 55364 },
  { event := event55438
    frameStart := 55364 },
  { event := event55439
    frameStart := 55364 }
]

def eventLeaf3465 : Array AnnotatedEvent := #[
  { event := event55440
    frameStart := 55364 },
  { event := event55441
    frameStart := 55364 },
  { event := event55442
    frameStart := 55364 },
  { event := event55443
    frameStart := 55364 },
  { event := event55444
    frameStart := 55364 },
  { event := event55445
    frameStart := 55364 },
  { event := event55446
    frameStart := 55364 },
  { event := event55447
    frameStart := 55364 },
  { event := event55448
    frameStart := 55364 },
  { event := event55449
    frameStart := 55364 },
  { event := event55450
    frameStart := 55364 },
  { event := event55451
    frameStart := 55364 },
  { event := event55452
    frameStart := 55364 },
  { event := event55453
    frameStart := 55364 },
  { event := event55454
    frameStart := 55364 },
  { event := event55455
    frameStart := 55364 }
]

def eventLeaf3466 : Array AnnotatedEvent := #[
  { event := event55456
    frameStart := 55364 },
  { event := event55457
    frameStart := 55364 },
  { event := event55458
    frameStart := 55364 },
  { event := event55459
    frameStart := 55364 },
  { event := event55460
    frameStart := 55364 },
  { event := event55461
    frameStart := 55364 },
  { event := event55462
    frameStart := 55364 },
  { event := event55463
    frameStart := 55364 },
  { event := event55464
    frameStart := 55364 },
  { event := event55465
    frameStart := 55364 },
  { event := event55466
    frameStart := 55364 },
  { event := event55467
    frameStart := 55364 },
  { event := event55468
    frameStart := 0 },
  { event := event55469
    frameStart := 0 },
  { event := event55470
    frameStart := 0 },
  { event := event55471
    frameStart := 0 }
]

def eventLeaf3467 : Array AnnotatedEvent := #[
  { event := event55472
    frameStart := 0 },
  { event := event55473
    frameStart := 0 },
  { event := event55474
    frameStart := 0 },
  { event := event55475
    frameStart := 0 },
  { event := event55476
    frameStart := 0 },
  { event := event55477
    frameStart := 0 },
  { event := event55478
    frameStart := 0 },
  { event := event55479
    frameStart := 0 },
  { event := event55480
    frameStart := 0 },
  { event := event55481
    frameStart := 0 },
  { event := event55482
    frameStart := 0 },
  { event := event55483
    frameStart := 0 },
  { event := event55484
    frameStart := 0 },
  { event := event55485
    frameStart := 0 },
  { event := event55486
    frameStart := 0 },
  { event := event55487
    frameStart := 0 }
]

def eventLeaf3468 : Array AnnotatedEvent := #[
  { event := event55488
    frameStart := 0 },
  { event := event55489
    frameStart := 0 },
  { event := event55490
    frameStart := 0 },
  { event := event55491
    frameStart := 0 },
  { event := event55492
    frameStart := 0 },
  { event := event55493
    frameStart := 0 },
  { event := event55494
    frameStart := 0 },
  { event := event55495
    frameStart := 0 },
  { event := event55496
    frameStart := 0 },
  { event := event55497
    frameStart := 0 },
  { event := event55498
    frameStart := 0 },
  { event := event55499
    frameStart := 0 },
  { event := event55500
    frameStart := 0 },
  { event := event55501
    frameStart := 0 },
  { event := event55502
    frameStart := 0 },
  { event := event55503
    frameStart := 0 }
]

def eventLeaf3469 : Array AnnotatedEvent := #[
  { event := event55504
    frameStart := 0 },
  { event := event55505
    frameStart := 0 },
  { event := event55506
    frameStart := 0 },
  { event := event55507
    frameStart := 0 },
  { event := event55508
    frameStart := 0 },
  { event := event55509
    frameStart := 0 },
  { event := event55510
    frameStart := 0 },
  { event := event55511
    frameStart := 0 },
  { event := event55512
    frameStart := 0 },
  { event := event55513
    frameStart := 0 },
  { event := event55514
    frameStart := 0 },
  { event := event55515
    frameStart := 0 },
  { event := event55516
    frameStart := 0 },
  { event := event55517
    frameStart := 0 },
  { event := event55518
    frameStart := 0 },
  { event := event55519
    frameStart := 0 }
]

def eventLeaf3470 : Array AnnotatedEvent := #[
  { event := event55520
    frameStart := 0 },
  { event := event55521
    frameStart := 0 },
  { event := event55522
    frameStart := 0 },
  { event := event55523
    frameStart := 0 },
  { event := event55524
    frameStart := 0 },
  { event := event55525
    frameStart := 0 },
  { event := event55526
    frameStart := 0 },
  { event := event55527
    frameStart := 0 },
  { event := event55528
    frameStart := 0 },
  { event := event55529
    frameStart := 0 },
  { event := event55530
    frameStart := 0 },
  { event := event55531
    frameStart := 0 },
  { event := event55532
    frameStart := 0 },
  { event := event55533
    frameStart := 0 },
  { event := event55534
    frameStart := 0 },
  { event := event55535
    frameStart := 0 }
]

def eventLeaf3471 : Array AnnotatedEvent := #[
  { event := event55536
    frameStart := 0 },
  { event := event55537
    frameStart := 0 },
  { event := event55538
    frameStart := 0 },
  { event := event55539
    frameStart := 0 },
  { event := event55540
    frameStart := 0 },
  { event := event55541
    frameStart := 0 },
  { event := event55542
    frameStart := 0 },
  { event := event55543
    frameStart := 0 },
  { event := event55544
    frameStart := 0 },
  { event := event55545
    frameStart := 0 },
  { event := event55546
    frameStart := 0 },
  { event := event55547
    frameStart := 0 },
  { event := event55548
    frameStart := 0 },
  { event := event55549
    frameStart := 0 },
  { event := event55550
    frameStart := 0 },
  { event := event55551
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events216
