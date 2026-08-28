import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events103

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event26368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.product (.predecessor 0 26366 .coefficient) (.predecessor 1 26367 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩) [⟨.result 26365 .coefficient, true, some 1⟩, ⟨.result 26362 .coefficient, true, some 1⟩])

def event26370 : Event := .survivorFold (1) 26369

def exact26371RawTerms : List Term := []

theorem exact26371RawTermsValid :
    exact26371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26371 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14235⟩⟩) exact26371RawTerms (.finite 324) 26368 (.finite 324) (some (26369))

def event26372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14236⟩⟩) 0 ⟨14235⟩ 26371

def event26373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.identity (.predecessor 0 26372 .coefficient))

def event26374 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.finite 324)

def event26375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19540⟩⟩) 0 ⟨14236⟩ 26374

def event26376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19540⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact26377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩, (1)⟩]

theorem exact26377RawTermsValid :
    exact26377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19540⟩⟩) exact26377RawTerms (.finite 136065468) 26376 .exactZero (none)

def event26378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact26379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact26379RawTermsValid :
    exact26379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact26379RawTerms .large 26378 .exactZero (none)

def event26380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19541⟩⟩) 0 ⟨6⟩ 26379

def event26381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19541⟩⟩) 1 ⟨19540⟩ 26377

def event26382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19541⟩⟩) (.product (.predecessor 0 26380 .coefficient) (.predecessor 1 26381 .coefficient) (⟨false, false, none, none, none⟩))

def event26383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19541⟩⟩, .operator (⟨26379, 0⟩, ⟨26377, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩, (1)⟩)

def exact26384RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩, (1)⟩]

theorem exact26384RawTermsValid :
    exact26384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26384 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19541⟩⟩) exact26384RawTerms .large 26382 .exactZero (none)

def event26385 : Event := .preFoldPolynomial 26384 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩, (1)⟩] .exactZero none

def exact26386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩, (1)⟩]

def event26386 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19541⟩⟩) 26385 exact26386RawTerms .large 26382 .exactZero (none)

def event26387 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26085⟩⟩)

def event26388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26391 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26393 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26395 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26395

def event26397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26393

def event26398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26396 .coefficient) (.value (.predecessor 1 26397 .coefficient)))

def event26399 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26399

def event26401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26391

def event26402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26400 .coefficient, .predecessor 1 26401 .coefficient])

def event26403 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26403

def event26405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26389

def event26406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26405 .coefficient))

def event26407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11481⟩⟩) 0 ⟨5554⟩ 26407

def event26409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11481⟩⟩) (.authority (.programFamilyFact))

def exact26410RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩], []⟩, (1)⟩]

theorem exact26410RawTermsValid :
    exact26410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11481⟩⟩) exact26410RawTerms (.finite 18) 26409 .exactZero (none)

def event26411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14234⟩⟩) 0 ⟨5554⟩ 26407

def event26412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14234⟩⟩) (.authority (.programFamilyFact))

def exact26413RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26413RawTermsValid :
    exact26413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14234⟩⟩) exact26413RawTerms (.finite 18) 26412 .exactZero (none)

def event26414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 0 ⟨14234⟩ 26413

def event26415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 1 ⟨11481⟩ 26410

def event26416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.product (.predecessor 0 26414 .coefficient) (.predecessor 1 26415 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26417 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14235⟩⟩, .operator (⟨26413, 0⟩, ⟨26410, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩)

def exact26418RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26418RawTermsValid :
    exact26418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14235⟩⟩) exact26418RawTerms (.finite 324) 26416 .exactZero (none)

def event26419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14236⟩⟩) 0 ⟨14235⟩ 26418

def event26420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.identity (.predecessor 0 26419 .coefficient))

def event26421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.finite 324)

def event26422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23589⟩⟩) 0 ⟨14236⟩ 26421

def event26423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23589⟩⟩) (.authority (.programFamilyFact))

def event26424 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23589⟩⟩) (.finite 3720)

def event26425 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event26426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23590⟩⟩) 0 ⟨6689⟩ 26425

def event26427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23590⟩⟩) 1 ⟨23589⟩ 26424

def event26428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23590⟩⟩) (.authority (.operator))

def exact26429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩]

theorem exact26429RawTermsValid :
    exact26429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23590⟩⟩) exact26429RawTerms .large 26428 .exactZero (none)

def event26430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26081⟩⟩) 0 ⟨23590⟩ 26429

def event26431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26081⟩⟩) (.authority (.operator))

def exact26432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (1)⟩]

theorem exact26432RawTermsValid :
    exact26432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26432 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26081⟩⟩) exact26432RawTerms (.finite 8192) 26431 .exactZero (none)

def event26433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event26434 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event26435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14326⟩⟩) 0 ⟨14236⟩ 26421

def event26436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14326⟩⟩) 1 ⟨110⟩ 26434

def event26437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14326⟩⟩) (.sum [.predecessor 0 26435 .coefficient, .predecessor 1 26436 .coefficient])

def event26438 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14326⟩⟩) (.finite 324)

def event26439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14327⟩⟩) 0 ⟨14326⟩ 26438

def event26440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14327⟩⟩) (.identity (.predecessor 0 26439 .coefficient))

def exact26441RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26441RawTermsValid :
    exact26441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26441 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14327⟩⟩) exact26441RawTerms (.finite 324) 26440 .exactZero (none)

def event26442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact26443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26443RawTermsValid :
    exact26443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact26443RawTerms .large 26442 .exactZero (none)

def event26444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14328⟩⟩) 0 ⟨6544⟩ 26443

def event26445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14328⟩⟩) 1 ⟨14327⟩ 26441

def event26446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14328⟩⟩) (.product (.predecessor 0 26444 .coefficient) (.predecessor 1 26445 .coefficient) (⟨false, false, none, none, none⟩))

def event26447 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14328⟩⟩, .operator (⟨26443, 0⟩, ⟨26441, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26448RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26448RawTermsValid :
    exact26448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26448 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14328⟩⟩) exact26448RawTerms .large 26446 .exactZero (none)

def event26449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event26450 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event26451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 26425

def event26452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact26453RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact26453RawTermsValid :
    exact26453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact26453RawTerms .large 26452 .exactZero (none)

def event26454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6779⟩⟩) 0 ⟨6757⟩ 26453

def event26455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6779⟩⟩) (.identity (.predecessor 0 26454 .coefficient))

def exact26456RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact26456RawTermsValid :
    exact26456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6779⟩⟩) exact26456RawTerms .large 26455 .exactZero (none)

def event26457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7852⟩⟩) 0 ⟨6779⟩ 26456

def event26458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7852⟩⟩) (.authority (.operator))

def exact26459RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact26459RawTermsValid :
    exact26459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7852⟩⟩) exact26459RawTerms (.finite 8192) 26458 .exactZero (none)

def event26460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 0 ⟨7852⟩ 26459

def event26461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 1 ⟨2348⟩ 26450

def event26462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7853⟩⟩) (.scale (.predecessor 0 26460 .coefficient) (.value (.predecessor 1 26461 .coefficient)))

def exact26463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact26463RawTermsValid :
    exact26463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26463 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7853⟩⟩) exact26463RawTerms (.finite 8192) 26462 .exactZero (none)

def event26464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6759⟩⟩) 0 ⟨6757⟩ 26453

def event26465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6759⟩⟩) (.identity (.predecessor 0 26464 .coefficient))

def exact26466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact26466RawTermsValid :
    exact26466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26466 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6759⟩⟩) exact26466RawTerms .large 26465 .exactZero (none)

def event26467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 0 ⟨6759⟩ 26466

def event26468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 1 ⟨7853⟩ 26463

def event26469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7854⟩⟩) (.product (.predecessor 0 26467 .coefficient) (.predecessor 1 26468 .coefficient) (⟨false, false, none, none, none⟩))

def event26470 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7854⟩⟩, .operator (⟨26466, 0⟩, ⟨26463, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact26471RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact26471RawTermsValid :
    exact26471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7854⟩⟩) exact26471RawTerms .large 26469 .exactZero (none)

def event26472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14329⟩⟩) 0 ⟨7854⟩ 26471

def event26473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14329⟩⟩) 1 ⟨14328⟩ 26448

def event26474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14329⟩⟩) (.sum [.predecessor 0 26472 .coefficient, .predecessor 1 26473 .coefficient])

def exact26475RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26475RawTermsValid :
    exact26475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26475 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14329⟩⟩) exact26475RawTerms .large 26474 .exactZero (none)

def event26476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26084⟩⟩) 0 ⟨14329⟩ 26475

def event26477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26084⟩⟩) 1 ⟨26081⟩ 26432

def event26478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26084⟩⟩) (.product (.predecessor 0 26476 .coefficient) (.predecessor 1 26477 .coefficient) (⟨false, false, none, none, none⟩))

def event26479 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26084⟩⟩, .operator (⟨26475, 0⟩, ⟨26432, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (1)⟩)

def event26480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26084⟩⟩, .operator (⟨26475, 1⟩, ⟨26432, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩)

def event26481 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26084⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26081⟩⟩) ⟨23590⟩ 26429)

def event26482 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26084⟩⟩, .relation 26481 0, ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (-1)⟩)

def exact26483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (-1)⟩]

theorem exact26483RawTermsValid :
    exact26483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26084⟩⟩) exact26483RawTerms .large 26478 .exactZero (none)

def event26484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15952⟩⟩) 0 ⟨14236⟩ 26421

def event26485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15952⟩⟩) (.authority (.programFamilyFact))

def exact26486RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], []⟩, (1)⟩]

theorem exact26486RawTermsValid :
    exact26486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26486 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15952⟩⟩) exact26486RawTerms (.finite 18) 26485 .exactZero (none)

def event26487 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15954⟩⟩) 0 ⟨6544⟩ 26443

def event26488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15954⟩⟩) 1 ⟨15952⟩ 26486

def event26489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15954⟩⟩) (.product (.predecessor 0 26487 .coefficient) (.predecessor 1 26488 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26490 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15954⟩⟩, .operator (⟨26443, 0⟩, ⟨26486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26491RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26491RawTermsValid :
    exact26491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15954⟩⟩) exact26491RawTerms .large 26489 .exactZero (none)

def event26492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 26425

def event26493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact26494RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact26494RawTermsValid :
    exact26494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact26494RawTerms .large 26493 .exactZero (none)

def event26495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15955⟩⟩) 0 ⟨6697⟩ 26494

def event26496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15955⟩⟩) 1 ⟨15954⟩ 26491

def event26497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15955⟩⟩) (.sum [.predecessor 0 26495 .coefficient, .predecessor 1 26496 .coefficient])

def exact26498RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26498RawTermsValid :
    exact26498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26498 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15955⟩⟩) exact26498RawTerms .large 26497 .exactZero (none)

def event26499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26085⟩⟩) 0 ⟨15955⟩ 26498

def event26500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26085⟩⟩) 1 ⟨26084⟩ 26483

def event26501 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26085⟩⟩) (.sum [.predecessor 0 26499 .coefficient, .predecessor 1 26500 .coefficient])

def exact26502RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26502RawTermsValid :
    exact26502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26085⟩⟩) exact26502RawTerms .large 26501 .exactZero (none)

def event26503 : Event := .preFoldPolynomial 26502 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact26504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event26504 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26085⟩⟩) 26503 exact26504RawTerms .large 26501 .exactZero (none)

def event26505 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14236⟩⟩) ⟨⟨110⟩, ⟨15⟩, ⟨109⟩⟩ ⟨26339, 26505⟩

def event26506 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19543⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (1) 0 2 (.universal 26505 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (none) 26504)

def event26507 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19543⟩⟩, .relation 26506 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩)

def event26508 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19543⟩⟩, .relation 26506 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩)

def event26509 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19543⟩⟩, .relation 26506 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩)

def event26510 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19543⟩⟩, .relation 26506 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact26511RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26511RawTermsValid :
    exact26511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26511 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19543⟩⟩) exact26511RawTerms .large 26335 (.finite 1811303510016) (some (26337))

def event26512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26083⟩⟩) 0 ⟨19543⟩ 26511

def event26513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26083⟩⟩) 1 ⟨26082⟩ 26325

def event26514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26083⟩⟩) (.sum [.predecessor 0 26512 .coefficient, .predecessor 1 26513 .coefficient])

def event26515 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26083⟩⟩, .operator (⟨26511, 2⟩, ⟨26325, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩, (-1)⟩)

def event26516 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26083⟩⟩, .operator (⟨26511, 1⟩, ⟨26325, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩, (1)⟩)

def event26517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26083⟩⟩) (.sum [.result 26511 .summary, .result 26325 .summary])

def exact26518RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26518RawTermsValid :
    exact26518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26518 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26083⟩⟩) exact26518RawTerms .large 26514 (.finite 352060719116288) (some (26517))

def event26519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27907⟩⟩) 0 ⟨26083⟩ 26518

def event26520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27907⟩⟩) 1 ⟨27905⟩ 26241

def event26521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27907⟩⟩) (.product (.predecessor 0 26519 .coefficient) (.predecessor 1 26520 .coefficient) (⟨false, false, none, none, none⟩))

def event26522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27907⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩) [⟨.result 26241 .coefficient, false, none⟩])

def event26523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27907⟩⟩) (.product (.result 26518 .summary) (.transfer 26522) (⟨false, false, none, none, none⟩))

def event26524 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27907⟩⟩, .operator (⟨26518, 0⟩, ⟨26241, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩)

def event26525 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27907⟩⟩, .operator (⟨26518, 1⟩, ⟨26241, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩)

def event26526 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27907⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27905⟩⟩) ⟨24171⟩ 26238)

def event26527 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27907⟩⟩, .relation 26526 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (-1)⟩)

def exact26528RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (-1)⟩]

theorem exact26528RawTermsValid :
    exact26528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26528 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27907⟩⟩) exact26528RawTerms .large 26521 (.finite 1292068472128282820608) (some (26523))

def event26529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21412⟩⟩) 0 ⟨15953⟩ 1089

def event26530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21412⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact26531RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩]

theorem exact26531RawTermsValid :
    exact26531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21412⟩⟩) exact26531RawTerms (.finite 136065468) 26530 .exactZero (none)

def event26532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21414⟩⟩) 0 ⟨21412⟩ 26531

def event26533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21414⟩⟩) 1 ⟨2348⟩ 4

def event26534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21414⟩⟩) (.scale (.predecessor 0 26532 .coefficient) (.value (.predecessor 1 26533 .coefficient)))

def exact26535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩]

theorem exact26535RawTermsValid :
    exact26535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21414⟩⟩) exact26535RawTerms (.finite 136065468) 26534 .exactZero (none)

def event26536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21415⟩⟩) 0 ⟨5559⟩ 21512

def event26537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21415⟩⟩) 1 ⟨21414⟩ 26535

def event26538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21415⟩⟩) (.product (.predecessor 0 26536 .coefficient) (.predecessor 1 26537 .coefficient) (⟨false, false, none, none, none⟩))

def event26539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21415⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩) [⟨.result 26531 .coefficient, false, none⟩])

def event26540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21415⟩⟩) (.product (.result 21512 .summary) (.transfer 26539) (⟨false, false, none, none, none⟩))

def event26541 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21415⟩⟩, .operator (⟨21512, 0⟩, ⟨26535, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩)

def event26542 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21413⟩⟩)

def event26543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26544 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26546 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26548 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26550 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26550

def event26552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26548

def event26553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26551 .coefficient) (.value (.predecessor 1 26552 .coefficient)))

def event26554 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26554

def event26556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26546

def event26557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26555 .coefficient, .predecessor 1 26556 .coefficient])

def event26558 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26558

def event26560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26544

def event26561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26560 .coefficient))

def event26562 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11481⟩⟩) 0 ⟨5554⟩ 26562

def event26564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11481⟩⟩) (.authority (.programFamilyFact))

def exact26565RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩], []⟩, (1)⟩]

theorem exact26565RawTermsValid :
    exact26565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26565 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11481⟩⟩) exact26565RawTerms (.finite 18) 26564 .exactZero (none)

def event26566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14234⟩⟩) 0 ⟨5554⟩ 26562

def event26567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14234⟩⟩) (.authority (.programFamilyFact))

def exact26568RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26568RawTermsValid :
    exact26568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26568 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14234⟩⟩) exact26568RawTerms (.finite 18) 26567 .exactZero (none)

def event26569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 0 ⟨14234⟩ 26568

def event26570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 1 ⟨11481⟩ 26565

def event26571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.product (.predecessor 0 26569 .coefficient) (.predecessor 1 26570 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩) [⟨.result 26568 .coefficient, true, some 1⟩, ⟨.result 26565 .coefficient, true, some 1⟩])

def event26573 : Event := .survivorFold (1) 26572

def exact26574RawTerms : List Term := []

theorem exact26574RawTermsValid :
    exact26574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14235⟩⟩) exact26574RawTerms (.finite 324) 26571 (.finite 324) (some (26572))

def event26575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14236⟩⟩) 0 ⟨14235⟩ 26574

def event26576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.identity (.predecessor 0 26575 .coefficient))

def event26577 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.finite 324)

def event26578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15952⟩⟩) 0 ⟨14236⟩ 26577

def event26579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15952⟩⟩) (.authority (.programFamilyFact))

def exact26580RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], []⟩, (1)⟩]

theorem exact26580RawTermsValid :
    exact26580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15952⟩⟩) exact26580RawTerms (.finite 18) 26579 .exactZero (none)

def event26581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15953⟩⟩) 0 ⟨15952⟩ 26580

def event26582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15953⟩⟩) (.identity (.predecessor 0 26581 .coefficient))

def event26583 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15953⟩⟩) (.finite 18)

def event26584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21412⟩⟩) 0 ⟨15953⟩ 26583

def event26585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21412⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact26586RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩]

theorem exact26586RawTermsValid :
    exact26586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26586 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21412⟩⟩) exact26586RawTerms (.finite 136065468) 26585 .exactZero (none)

def event26587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact26588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact26588RawTermsValid :
    exact26588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact26588RawTerms .large 26587 .exactZero (none)

def event26589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21413⟩⟩) 0 ⟨6⟩ 26588

def event26590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21413⟩⟩) 1 ⟨21412⟩ 26586

def event26591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21413⟩⟩) (.product (.predecessor 0 26589 .coefficient) (.predecessor 1 26590 .coefficient) (⟨false, false, none, none, none⟩))

def event26592 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21413⟩⟩, .operator (⟨26588, 0⟩, ⟨26586, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩)

def exact26593RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩]

theorem exact26593RawTermsValid :
    exact26593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26593 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21413⟩⟩) exact26593RawTerms .large 26591 .exactZero (none)

def event26594 : Event := .preFoldPolynomial 26593 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩] .exactZero none

def exact26595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩, (1)⟩]

def event26595 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21413⟩⟩) 26594 exact26595RawTerms .large 26591 .exactZero (none)

def event26596 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27910⟩⟩)

def event26597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26598 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26601 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26602 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26604

def event26606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26602

def event26607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26605 .coefficient) (.value (.predecessor 1 26606 .coefficient)))

def event26608 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26608

def event26610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26600

def event26611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26609 .coefficient, .predecessor 1 26610 .coefficient])

def event26612 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26612

def event26614 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26598

def event26615 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26614 .coefficient))

def event26616 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11481⟩⟩) 0 ⟨5554⟩ 26616

def event26618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11481⟩⟩) (.authority (.programFamilyFact))

def exact26619RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩], []⟩, (1)⟩]

theorem exact26619RawTermsValid :
    exact26619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26619 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11481⟩⟩) exact26619RawTerms (.finite 18) 26618 .exactZero (none)

def event26620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14234⟩⟩) 0 ⟨5554⟩ 26616

def event26621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14234⟩⟩) (.authority (.programFamilyFact))

def exact26622RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26622RawTermsValid :
    exact26622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26622 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14234⟩⟩) exact26622RawTerms (.finite 18) 26621 .exactZero (none)

def event26623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 0 ⟨14234⟩ 26622

def eventLeaf1648 : Array AnnotatedEvent := #[
  { event := event26368
    frameStart := 26339 },
  { event := event26369
    frameStart := 26339 },
  { event := event26370
    frameStart := 26339 },
  { event := event26371
    frameStart := 26339 },
  { event := event26372
    frameStart := 26339 },
  { event := event26373
    frameStart := 26339 },
  { event := event26374
    frameStart := 26339 },
  { event := event26375
    frameStart := 26339 },
  { event := event26376
    frameStart := 26339 },
  { event := event26377
    frameStart := 26339 },
  { event := event26378
    frameStart := 26339 },
  { event := event26379
    frameStart := 26339 },
  { event := event26380
    frameStart := 26339 },
  { event := event26381
    frameStart := 26339 },
  { event := event26382
    frameStart := 26339 },
  { event := event26383
    frameStart := 26339 }
]

def eventLeaf1649 : Array AnnotatedEvent := #[
  { event := event26384
    frameStart := 26339 },
  { event := event26385
    frameStart := 26339 },
  { event := event26386
    frameStart := 26339 },
  { event := event26387
    frameStart := 26387 },
  { event := event26388
    frameStart := 26387 },
  { event := event26389
    frameStart := 26387 },
  { event := event26390
    frameStart := 26387 },
  { event := event26391
    frameStart := 26387 },
  { event := event26392
    frameStart := 26387 },
  { event := event26393
    frameStart := 26387 },
  { event := event26394
    frameStart := 26387 },
  { event := event26395
    frameStart := 26387 },
  { event := event26396
    frameStart := 26387 },
  { event := event26397
    frameStart := 26387 },
  { event := event26398
    frameStart := 26387 },
  { event := event26399
    frameStart := 26387 }
]

def eventLeaf1650 : Array AnnotatedEvent := #[
  { event := event26400
    frameStart := 26387 },
  { event := event26401
    frameStart := 26387 },
  { event := event26402
    frameStart := 26387 },
  { event := event26403
    frameStart := 26387 },
  { event := event26404
    frameStart := 26387 },
  { event := event26405
    frameStart := 26387 },
  { event := event26406
    frameStart := 26387 },
  { event := event26407
    frameStart := 26387 },
  { event := event26408
    frameStart := 26387 },
  { event := event26409
    frameStart := 26387 },
  { event := event26410
    frameStart := 26387 },
  { event := event26411
    frameStart := 26387 },
  { event := event26412
    frameStart := 26387 },
  { event := event26413
    frameStart := 26387 },
  { event := event26414
    frameStart := 26387 },
  { event := event26415
    frameStart := 26387 }
]

def eventLeaf1651 : Array AnnotatedEvent := #[
  { event := event26416
    frameStart := 26387 },
  { event := event26417
    frameStart := 26387 },
  { event := event26418
    frameStart := 26387 },
  { event := event26419
    frameStart := 26387 },
  { event := event26420
    frameStart := 26387 },
  { event := event26421
    frameStart := 26387 },
  { event := event26422
    frameStart := 26387 },
  { event := event26423
    frameStart := 26387 },
  { event := event26424
    frameStart := 26387 },
  { event := event26425
    frameStart := 26387 },
  { event := event26426
    frameStart := 26387 },
  { event := event26427
    frameStart := 26387 },
  { event := event26428
    frameStart := 26387 },
  { event := event26429
    frameStart := 26387 },
  { event := event26430
    frameStart := 26387 },
  { event := event26431
    frameStart := 26387 }
]

def eventLeaf1652 : Array AnnotatedEvent := #[
  { event := event26432
    frameStart := 26387 },
  { event := event26433
    frameStart := 26387 },
  { event := event26434
    frameStart := 26387 },
  { event := event26435
    frameStart := 26387 },
  { event := event26436
    frameStart := 26387 },
  { event := event26437
    frameStart := 26387 },
  { event := event26438
    frameStart := 26387 },
  { event := event26439
    frameStart := 26387 },
  { event := event26440
    frameStart := 26387 },
  { event := event26441
    frameStart := 26387 },
  { event := event26442
    frameStart := 26387 },
  { event := event26443
    frameStart := 26387 },
  { event := event26444
    frameStart := 26387 },
  { event := event26445
    frameStart := 26387 },
  { event := event26446
    frameStart := 26387 },
  { event := event26447
    frameStart := 26387 }
]

def eventLeaf1653 : Array AnnotatedEvent := #[
  { event := event26448
    frameStart := 26387 },
  { event := event26449
    frameStart := 26387 },
  { event := event26450
    frameStart := 26387 },
  { event := event26451
    frameStart := 26387 },
  { event := event26452
    frameStart := 26387 },
  { event := event26453
    frameStart := 26387 },
  { event := event26454
    frameStart := 26387 },
  { event := event26455
    frameStart := 26387 },
  { event := event26456
    frameStart := 26387 },
  { event := event26457
    frameStart := 26387 },
  { event := event26458
    frameStart := 26387 },
  { event := event26459
    frameStart := 26387 },
  { event := event26460
    frameStart := 26387 },
  { event := event26461
    frameStart := 26387 },
  { event := event26462
    frameStart := 26387 },
  { event := event26463
    frameStart := 26387 }
]

def eventLeaf1654 : Array AnnotatedEvent := #[
  { event := event26464
    frameStart := 26387 },
  { event := event26465
    frameStart := 26387 },
  { event := event26466
    frameStart := 26387 },
  { event := event26467
    frameStart := 26387 },
  { event := event26468
    frameStart := 26387 },
  { event := event26469
    frameStart := 26387 },
  { event := event26470
    frameStart := 26387 },
  { event := event26471
    frameStart := 26387 },
  { event := event26472
    frameStart := 26387 },
  { event := event26473
    frameStart := 26387 },
  { event := event26474
    frameStart := 26387 },
  { event := event26475
    frameStart := 26387 },
  { event := event26476
    frameStart := 26387 },
  { event := event26477
    frameStart := 26387 },
  { event := event26478
    frameStart := 26387 },
  { event := event26479
    frameStart := 26387 }
]

def eventLeaf1655 : Array AnnotatedEvent := #[
  { event := event26480
    frameStart := 26387 },
  { event := event26481
    frameStart := 26387 },
  { event := event26482
    frameStart := 26387 },
  { event := event26483
    frameStart := 26387 },
  { event := event26484
    frameStart := 26387 },
  { event := event26485
    frameStart := 26387 },
  { event := event26486
    frameStart := 26387 },
  { event := event26487
    frameStart := 26387 },
  { event := event26488
    frameStart := 26387 },
  { event := event26489
    frameStart := 26387 },
  { event := event26490
    frameStart := 26387 },
  { event := event26491
    frameStart := 26387 },
  { event := event26492
    frameStart := 26387 },
  { event := event26493
    frameStart := 26387 },
  { event := event26494
    frameStart := 26387 },
  { event := event26495
    frameStart := 26387 }
]

def eventLeaf1656 : Array AnnotatedEvent := #[
  { event := event26496
    frameStart := 26387 },
  { event := event26497
    frameStart := 26387 },
  { event := event26498
    frameStart := 26387 },
  { event := event26499
    frameStart := 26387 },
  { event := event26500
    frameStart := 26387 },
  { event := event26501
    frameStart := 26387 },
  { event := event26502
    frameStart := 26387 },
  { event := event26503
    frameStart := 26387 },
  { event := event26504
    frameStart := 26387 },
  { event := event26505
    frameStart := 0 },
  { event := event26506
    frameStart := 0 },
  { event := event26507
    frameStart := 0 },
  { event := event26508
    frameStart := 0 },
  { event := event26509
    frameStart := 0 },
  { event := event26510
    frameStart := 0 },
  { event := event26511
    frameStart := 0 }
]

def eventLeaf1657 : Array AnnotatedEvent := #[
  { event := event26512
    frameStart := 0 },
  { event := event26513
    frameStart := 0 },
  { event := event26514
    frameStart := 0 },
  { event := event26515
    frameStart := 0 },
  { event := event26516
    frameStart := 0 },
  { event := event26517
    frameStart := 0 },
  { event := event26518
    frameStart := 0 },
  { event := event26519
    frameStart := 0 },
  { event := event26520
    frameStart := 0 },
  { event := event26521
    frameStart := 0 },
  { event := event26522
    frameStart := 0 },
  { event := event26523
    frameStart := 0 },
  { event := event26524
    frameStart := 0 },
  { event := event26525
    frameStart := 0 },
  { event := event26526
    frameStart := 0 },
  { event := event26527
    frameStart := 0 }
]

def eventLeaf1658 : Array AnnotatedEvent := #[
  { event := event26528
    frameStart := 0 },
  { event := event26529
    frameStart := 0 },
  { event := event26530
    frameStart := 0 },
  { event := event26531
    frameStart := 0 },
  { event := event26532
    frameStart := 0 },
  { event := event26533
    frameStart := 0 },
  { event := event26534
    frameStart := 0 },
  { event := event26535
    frameStart := 0 },
  { event := event26536
    frameStart := 0 },
  { event := event26537
    frameStart := 0 },
  { event := event26538
    frameStart := 0 },
  { event := event26539
    frameStart := 0 },
  { event := event26540
    frameStart := 0 },
  { event := event26541
    frameStart := 0 },
  { event := event26542
    frameStart := 26542 },
  { event := event26543
    frameStart := 26542 }
]

def eventLeaf1659 : Array AnnotatedEvent := #[
  { event := event26544
    frameStart := 26542 },
  { event := event26545
    frameStart := 26542 },
  { event := event26546
    frameStart := 26542 },
  { event := event26547
    frameStart := 26542 },
  { event := event26548
    frameStart := 26542 },
  { event := event26549
    frameStart := 26542 },
  { event := event26550
    frameStart := 26542 },
  { event := event26551
    frameStart := 26542 },
  { event := event26552
    frameStart := 26542 },
  { event := event26553
    frameStart := 26542 },
  { event := event26554
    frameStart := 26542 },
  { event := event26555
    frameStart := 26542 },
  { event := event26556
    frameStart := 26542 },
  { event := event26557
    frameStart := 26542 },
  { event := event26558
    frameStart := 26542 },
  { event := event26559
    frameStart := 26542 }
]

def eventLeaf1660 : Array AnnotatedEvent := #[
  { event := event26560
    frameStart := 26542 },
  { event := event26561
    frameStart := 26542 },
  { event := event26562
    frameStart := 26542 },
  { event := event26563
    frameStart := 26542 },
  { event := event26564
    frameStart := 26542 },
  { event := event26565
    frameStart := 26542 },
  { event := event26566
    frameStart := 26542 },
  { event := event26567
    frameStart := 26542 },
  { event := event26568
    frameStart := 26542 },
  { event := event26569
    frameStart := 26542 },
  { event := event26570
    frameStart := 26542 },
  { event := event26571
    frameStart := 26542 },
  { event := event26572
    frameStart := 26542 },
  { event := event26573
    frameStart := 26542 },
  { event := event26574
    frameStart := 26542 },
  { event := event26575
    frameStart := 26542 }
]

def eventLeaf1661 : Array AnnotatedEvent := #[
  { event := event26576
    frameStart := 26542 },
  { event := event26577
    frameStart := 26542 },
  { event := event26578
    frameStart := 26542 },
  { event := event26579
    frameStart := 26542 },
  { event := event26580
    frameStart := 26542 },
  { event := event26581
    frameStart := 26542 },
  { event := event26582
    frameStart := 26542 },
  { event := event26583
    frameStart := 26542 },
  { event := event26584
    frameStart := 26542 },
  { event := event26585
    frameStart := 26542 },
  { event := event26586
    frameStart := 26542 },
  { event := event26587
    frameStart := 26542 },
  { event := event26588
    frameStart := 26542 },
  { event := event26589
    frameStart := 26542 },
  { event := event26590
    frameStart := 26542 },
  { event := event26591
    frameStart := 26542 }
]

def eventLeaf1662 : Array AnnotatedEvent := #[
  { event := event26592
    frameStart := 26542 },
  { event := event26593
    frameStart := 26542 },
  { event := event26594
    frameStart := 26542 },
  { event := event26595
    frameStart := 26542 },
  { event := event26596
    frameStart := 26596 },
  { event := event26597
    frameStart := 26596 },
  { event := event26598
    frameStart := 26596 },
  { event := event26599
    frameStart := 26596 },
  { event := event26600
    frameStart := 26596 },
  { event := event26601
    frameStart := 26596 },
  { event := event26602
    frameStart := 26596 },
  { event := event26603
    frameStart := 26596 },
  { event := event26604
    frameStart := 26596 },
  { event := event26605
    frameStart := 26596 },
  { event := event26606
    frameStart := 26596 },
  { event := event26607
    frameStart := 26596 }
]

def eventLeaf1663 : Array AnnotatedEvent := #[
  { event := event26608
    frameStart := 26596 },
  { event := event26609
    frameStart := 26596 },
  { event := event26610
    frameStart := 26596 },
  { event := event26611
    frameStart := 26596 },
  { event := event26612
    frameStart := 26596 },
  { event := event26613
    frameStart := 26596 },
  { event := event26614
    frameStart := 26596 },
  { event := event26615
    frameStart := 26596 },
  { event := event26616
    frameStart := 26596 },
  { event := event26617
    frameStart := 26596 },
  { event := event26618
    frameStart := 26596 },
  { event := event26619
    frameStart := 26596 },
  { event := event26620
    frameStart := 26596 },
  { event := event26621
    frameStart := 26596 },
  { event := event26622
    frameStart := 26596 },
  { event := event26623
    frameStart := 26596 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events103
