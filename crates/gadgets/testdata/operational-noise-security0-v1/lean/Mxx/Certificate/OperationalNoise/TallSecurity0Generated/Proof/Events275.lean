import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events275

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event70400 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27855⟩⟩, .operator (⟨70393, 1⟩, ⟨70116, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩)

def event70401 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27855⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27853⟩⟩) ⟨24159⟩ 70113)

def event70402 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27855⟩⟩, .relation 70401 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (-1)⟩)

def exact70403RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (-1)⟩]

theorem exact70403RawTermsValid :
    exact70403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27855⟩⟩) exact70403RawTerms .large 70396 (.finite 1292068472128282820608) (some (70398))

def event70404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21396⟩⟩) 0 ⟨15937⟩ 3333

def event70405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21396⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact70406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩]

theorem exact70406RawTermsValid :
    exact70406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21396⟩⟩) exact70406RawTerms (.finite 136065468) 70405 .exactZero (none)

def event70407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21398⟩⟩) 0 ⟨21396⟩ 70406

def event70408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21398⟩⟩) 1 ⟨2348⟩ 4

def event70409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21398⟩⟩) (.scale (.predecessor 0 70407 .coefficient) (.value (.predecessor 1 70408 .coefficient)))

def exact70410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩]

theorem exact70410RawTermsValid :
    exact70410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21398⟩⟩) exact70410RawTerms (.finite 136065468) 70409 .exactZero (none)

def event70411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21399⟩⟩) 0 ⟨5535⟩ 65387

def event70412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21399⟩⟩) 1 ⟨21398⟩ 70410

def event70413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21399⟩⟩) (.product (.predecessor 0 70411 .coefficient) (.predecessor 1 70412 .coefficient) (⟨false, false, none, none, none⟩))

def event70414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21399⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩) [⟨.result 70406 .coefficient, false, none⟩])

def event70415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21399⟩⟩) (.product (.result 65387 .summary) (.transfer 70414) (⟨false, false, none, none, none⟩))

def event70416 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21399⟩⟩, .operator (⟨65387, 0⟩, ⟨70410, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩)

def event70417 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21397⟩⟩)

def event70418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70419 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70423 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70425 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70425

def event70427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70423

def event70428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70426 .coefficient) (.value (.predecessor 1 70427 .coefficient)))

def event70429 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70429

def event70431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70421

def event70432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70430 .coefficient, .predecessor 1 70431 .coefficient])

def event70433 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70433

def event70435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70419

def event70436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70435 .coefficient))

def event70437 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70438 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 70437

def event70439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact70440RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact70440RawTermsValid :
    exact70440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70440 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact70440RawTerms (.finite 18) 70439 .exactZero (none)

def event70441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 70437

def event70442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact70443RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70443RawTermsValid :
    exact70443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact70443RawTerms (.finite 18) 70442 .exactZero (none)

def event70444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 70443

def event70445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 70440

def event70446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 70444 .coefficient) (.predecessor 1 70445 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩) [⟨.result 70443 .coefficient, true, some 1⟩, ⟨.result 70440 .coefficient, true, some 1⟩])

def event70448 : Event := .survivorFold (1) 70447

def exact70449RawTerms : List Term := []

theorem exact70449RawTermsValid :
    exact70449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70449 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact70449RawTerms (.finite 324) 70446 (.finite 324) (some (70447))

def event70450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 70449

def event70451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 70450 .coefficient))

def event70452 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event70453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 70452

def event70454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact70455RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact70455RawTermsValid :
    exact70455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70455 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact70455RawTerms (.finite 18) 70454 .exactZero (none)

def event70456 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15937⟩⟩) 0 ⟨15936⟩ 70455

def event70457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.identity (.predecessor 0 70456 .coefficient))

def event70458 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.finite 18)

def event70459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21396⟩⟩) 0 ⟨15937⟩ 70458

def event70460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21396⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact70461RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩]

theorem exact70461RawTermsValid :
    exact70461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21396⟩⟩) exact70461RawTerms (.finite 136065468) 70460 .exactZero (none)

def event70462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact70463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact70463RawTermsValid :
    exact70463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70463 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact70463RawTerms .large 70462 .exactZero (none)

def event70464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21397⟩⟩) 0 ⟨6⟩ 70463

def event70465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21397⟩⟩) 1 ⟨21396⟩ 70461

def event70466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21397⟩⟩) (.product (.predecessor 0 70464 .coefficient) (.predecessor 1 70465 .coefficient) (⟨false, false, none, none, none⟩))

def event70467 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21397⟩⟩, .operator (⟨70463, 0⟩, ⟨70461, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩)

def exact70468RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩]

theorem exact70468RawTermsValid :
    exact70468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70468 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21397⟩⟩) exact70468RawTerms .large 70466 .exactZero (none)

def event70469 : Event := .preFoldPolynomial 70468 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩] .exactZero none

def exact70470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩, (1)⟩]

def event70470 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21397⟩⟩) 70469 exact70470RawTerms .large 70466 .exactZero (none)

def event70471 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27858⟩⟩)

def event70472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70473 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70475 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70477 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70479 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70479

def event70481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70477

def event70482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70480 .coefficient) (.value (.predecessor 1 70481 .coefficient)))

def event70483 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70483

def event70485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70475

def event70486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70484 .coefficient, .predecessor 1 70485 .coefficient])

def event70487 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70487

def event70489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70473

def event70490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70489 .coefficient))

def event70491 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 70491

def event70493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact70494RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact70494RawTermsValid :
    exact70494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact70494RawTerms (.finite 18) 70493 .exactZero (none)

def event70495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 70491

def event70496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact70497RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70497RawTermsValid :
    exact70497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact70497RawTerms (.finite 18) 70496 .exactZero (none)

def event70498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 70497

def event70499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 70494

def event70500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 70498 .coefficient) (.predecessor 1 70499 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14199⟩⟩, .operator (⟨70497, 0⟩, ⟨70494, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩)

def exact70502RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact70502RawTermsValid :
    exact70502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact70502RawTerms (.finite 324) 70500 .exactZero (none)

def event70503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 70502

def event70504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 70503 .coefficient))

def event70505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event70506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 70505

def event70507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact70508RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact70508RawTermsValid :
    exact70508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70508 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact70508RawTerms (.finite 18) 70507 .exactZero (none)

def event70509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15937⟩⟩) 0 ⟨15936⟩ 70508

def event70510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.identity (.predecessor 0 70509 .coefficient))

def event70511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.finite 18)

def event70512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24157⟩⟩) 0 ⟨15937⟩ 70511

def event70513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.authority (.programFamilyFact))

def event70514 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.finite 3720)

def event70515 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event70516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24159⟩⟩) 0 ⟨6689⟩ 70515

def event70517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24159⟩⟩) 1 ⟨24157⟩ 70514

def event70518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24159⟩⟩) (.authority (.operator))

def exact70519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩]

theorem exact70519RawTermsValid :
    exact70519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70519 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24159⟩⟩) exact70519RawTerms .large 70518 .exactZero (none)

def event70520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27853⟩⟩) 0 ⟨24159⟩ 70519

def event70521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27853⟩⟩) (.authority (.operator))

def exact70522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩]

theorem exact70522RawTermsValid :
    exact70522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27853⟩⟩) exact70522RawTerms (.finite 8192) 70521 .exactZero (none)

def event70523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event70524 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event70525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16011⟩⟩) 0 ⟨15937⟩ 70511

def event70526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16011⟩⟩) 1 ⟨110⟩ 70524

def event70527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16011⟩⟩) (.sum [.predecessor 0 70525 .coefficient, .predecessor 1 70526 .coefficient])

def event70528 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16011⟩⟩) (.finite 18)

def event70529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16012⟩⟩) 0 ⟨16011⟩ 70528

def event70530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16012⟩⟩) (.identity (.predecessor 0 70529 .coefficient))

def exact70531RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact70531RawTermsValid :
    exact70531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16012⟩⟩) exact70531RawTerms (.finite 18) 70530 .exactZero (none)

def event70532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact70533RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70533RawTermsValid :
    exact70533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact70533RawTerms .large 70532 .exactZero (none)

def event70534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16013⟩⟩) 0 ⟨6544⟩ 70533

def event70535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16013⟩⟩) 1 ⟨16012⟩ 70531

def event70536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16013⟩⟩) (.product (.predecessor 0 70534 .coefficient) (.predecessor 1 70535 .coefficient) (⟨false, false, none, none, none⟩))

def event70537 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16013⟩⟩, .operator (⟨70533, 0⟩, ⟨70531, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70538RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70538RawTermsValid :
    exact70538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16013⟩⟩) exact70538RawTerms .large 70536 .exactZero (none)

def event70539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 70515

def event70540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact70541RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact70541RawTermsValid :
    exact70541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70541 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact70541RawTerms .large 70540 .exactZero (none)

def event70542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16014⟩⟩) 0 ⟨6697⟩ 70541

def event70543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16014⟩⟩) 1 ⟨16013⟩ 70538

def event70544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16014⟩⟩) (.sum [.predecessor 0 70542 .coefficient, .predecessor 1 70543 .coefficient])

def exact70545RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70545RawTermsValid :
    exact70545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16014⟩⟩) exact70545RawTerms .large 70544 .exactZero (none)

def event70546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27854⟩⟩) 0 ⟨16014⟩ 70545

def event70547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27854⟩⟩) 1 ⟨27853⟩ 70522

def event70548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27854⟩⟩) (.product (.predecessor 0 70546 .coefficient) (.predecessor 1 70547 .coefficient) (⟨false, false, none, none, none⟩))

def event70549 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27854⟩⟩, .operator (⟨70545, 0⟩, ⟨70522, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩)

def event70550 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27854⟩⟩, .operator (⟨70545, 1⟩, ⟨70522, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩)

def event70551 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27854⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27853⟩⟩) ⟨24159⟩ 70519)

def event70552 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27854⟩⟩, .relation 70551 0, ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (-1)⟩)

def exact70553RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (-1)⟩]

theorem exact70553RawTermsValid :
    exact70553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70553 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27854⟩⟩) exact70553RawTerms .large 70548 .exactZero (none)

def event70554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15983⟩⟩) 0 ⟨15937⟩ 70511

def event70555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15983⟩⟩) (.authority (.programFamilyFact))

def exact70556RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], []⟩, (1)⟩]

theorem exact70556RawTermsValid :
    exact70556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70556 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15983⟩⟩) exact70556RawTerms (.finite 61) 70555 .exactZero (none)

def event70557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15984⟩⟩) 0 ⟨6544⟩ 70533

def event70558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15984⟩⟩) 1 ⟨15983⟩ 70556

def event70559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15984⟩⟩) (.product (.predecessor 0 70557 .coefficient) (.predecessor 1 70558 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70560 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15984⟩⟩, .operator (⟨70533, 0⟩, ⟨70556, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70561RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70561RawTermsValid :
    exact70561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15984⟩⟩) exact70561RawTerms .large 70559 .exactZero (none)

def event70562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6723⟩⟩) 0 ⟨6689⟩ 70515

def event70563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6723⟩⟩) (.authority (.operator))

def exact70564RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩]

theorem exact70564RawTermsValid :
    exact70564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70564 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6723⟩⟩) exact70564RawTerms .large 70563 .exactZero (none)

def event70565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15985⟩⟩) 0 ⟨6723⟩ 70564

def event70566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15985⟩⟩) 1 ⟨15984⟩ 70561

def event70567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15985⟩⟩) (.sum [.predecessor 0 70565 .coefficient, .predecessor 1 70566 .coefficient])

def exact70568RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70568RawTermsValid :
    exact70568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70568 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15985⟩⟩) exact70568RawTerms .large 70567 .exactZero (none)

def event70569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27858⟩⟩) 0 ⟨15985⟩ 70568

def event70570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27858⟩⟩) 1 ⟨27854⟩ 70553

def event70571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27858⟩⟩) (.sum [.predecessor 0 70569 .coefficient, .predecessor 1 70570 .coefficient])

def exact70572RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70572RawTermsValid :
    exact70572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70572 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27858⟩⟩) exact70572RawTerms .large 70571 .exactZero (none)

def event70573 : Event := .preFoldPolynomial 70572 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact70574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event70574 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27858⟩⟩) 70573 exact70574RawTerms .large 70571 .exactZero (none)

def event70575 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15937⟩⟩) ⟨⟨136⟩, ⟨43⟩, ⟨109⟩⟩ ⟨70417, 70575⟩

def event70576 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21399⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩) (1) 0 2 (.universal 70575 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩) (none) 70574)

def event70577 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21399⟩⟩, .relation 70576 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩)

def event70578 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21399⟩⟩, .relation 70576 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩)

def event70579 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21399⟩⟩, .relation 70576 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩)

def event70580 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21399⟩⟩, .relation 70576 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact70581RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70581RawTermsValid :
    exact70581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21399⟩⟩) exact70581RawTerms .large 70413 (.finite 1811303510016) (some (70415))

def event70582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27856⟩⟩) 0 ⟨21399⟩ 70581

def event70583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27856⟩⟩) 1 ⟨27855⟩ 70403

def event70584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27856⟩⟩) (.sum [.predecessor 0 70582 .coefficient, .predecessor 1 70583 .coefficient])

def event70585 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27856⟩⟩, .operator (⟨70581, 0⟩, ⟨70403, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩)

def event70586 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27856⟩⟩, .operator (⟨70581, 2⟩, ⟨70403, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (-1)⟩)

def event70587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27856⟩⟩) (.sum [.result 70581 .summary, .result 70403 .summary])

def exact70588RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70588RawTermsValid :
    exact70588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27856⟩⟩) exact70588RawTerms .large 70584 (.finite 1292068473939586330624) (some (70587))

def event70589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24094⟩⟩) 0 ⟨15818⟩ 3356

def event70590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.authority (.programFamilyFact))

def event70591 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.finite 3720)

def event70592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24096⟩⟩) 0 ⟨6689⟩ 5477

def event70593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24096⟩⟩) 1 ⟨24094⟩ 70591

def event70594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24096⟩⟩) (.authority (.operator))

def exact70595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩]

theorem exact70595RawTermsValid :
    exact70595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70595 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24096⟩⟩) exact70595RawTerms .large 70594 .exactZero (none)

def event70596 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27636⟩⟩) 0 ⟨24096⟩ 70595

def event70597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27636⟩⟩) (.authority (.operator))

def exact70598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩]

theorem exact70598RawTermsValid :
    exact70598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70598 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27636⟩⟩) exact70598RawTerms (.finite 8192) 70597 .exactZero (none)

def event70599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23539⟩⟩) 0 ⟨13983⟩ 3350

def event70600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23539⟩⟩) (.authority (.programFamilyFact))

def event70601 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23539⟩⟩) (.finite 3720)

def event70602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23540⟩⟩) 0 ⟨6689⟩ 5477

def event70603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23540⟩⟩) 1 ⟨23539⟩ 70601

def event70604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23540⟩⟩) (.authority (.operator))

def exact70605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩]

theorem exact70605RawTermsValid :
    exact70605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70605 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23540⟩⟩) exact70605RawTerms .large 70604 .exactZero (none)

def event70606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25984⟩⟩) 0 ⟨23540⟩ 70605

def event70607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25984⟩⟩) (.authority (.operator))

def exact70608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩]

theorem exact70608RawTermsValid :
    exact70608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70608 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25984⟩⟩) exact70608RawTerms (.finite 8192) 70607 .exactZero (none)

def event70609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11382⟩⟩) 0 ⟨11381⟩ 3339

def event70610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11382⟩⟩) 1 ⟨6566⟩ 65295

def event70611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11382⟩⟩) (.tensor (.predecessor 0 70609 .coefficient) (.predecessor 1 70610 .coefficient) true false)

def event70612 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11382⟩⟩, .operator (⟨3339, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70613RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70613RawTermsValid :
    exact70613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70613 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11382⟩⟩) exact70613RawTerms .large 70611 .exactZero (none)

def event70614 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7196⟩⟩) 0 ⟨5533⟩ 65165

def event70615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7196⟩⟩) 1 ⟨6778⟩ 11983

def event70616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7196⟩⟩) (.product (.predecessor 0 70614 .coefficient) (.predecessor 1 70615 .coefficient) (⟨false, false, none, none, none⟩))

def event70617 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7196⟩⟩, .operator (⟨65165, 0⟩, ⟨11983, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact70618RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact70618RawTermsValid :
    exact70618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70618 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7196⟩⟩) exact70618RawTerms .large 70616 .exactZero (none)

def event70619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11383⟩⟩) 0 ⟨7196⟩ 70618

def event70620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11383⟩⟩) 1 ⟨11382⟩ 70613

def event70621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11383⟩⟩) (.sum [.predecessor 0 70619 .coefficient, .predecessor 1 70620 .coefficient])

def exact70622RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70622RawTermsValid :
    exact70622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70622 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11383⟩⟩) exact70622RawTerms .large 70621 .exactZero (none)

def event70623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11384⟩⟩) 0 ⟨11383⟩ 70622

def event70624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11384⟩⟩) 1 ⟨92⟩ 11975

def event70625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11384⟩⟩) (.sum [.predecessor 0 70623 .coefficient, .predecessor 1 70624 .coefficient])

def event70626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11384⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) [⟨.result 11975 .coefficient, false, none⟩])

def event70627 : Event := .survivorFold (1) 70626

def exact70628RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70628RawTermsValid :
    exact70628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11384⟩⟩) exact70628RawTerms .large 70625 (.finite 26) (some (70626))

def event70629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13984⟩⟩) 0 ⟨11384⟩ 70628

def event70630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13984⟩⟩) 1 ⟨13981⟩ 3342

def event70631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13984⟩⟩) (.product (.predecessor 0 70629 .coefficient) (.predecessor 1 70630 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13984⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩) [⟨.result 3342 .coefficient, true, some 1⟩])

def event70633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13984⟩⟩) (.product (.result 70628 .summary) (.transfer 70632) (⟨false, false, none, none, none⟩))

def event70634 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13984⟩⟩, .operator (⟨70628, 1⟩, ⟨3342, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event70635 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13984⟩⟩, .operator (⟨70628, 0⟩, ⟨3342, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact70636RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact70636RawTermsValid :
    exact70636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13984⟩⟩) exact70636RawTerms .large 70631 (.finite 13312) (some (70633))

def event70637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13985⟩⟩) 0 ⟨13981⟩ 3342

def event70638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13985⟩⟩) 1 ⟨6566⟩ 65295

def event70639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13985⟩⟩) (.tensor (.predecessor 0 70637 .coefficient) (.predecessor 1 70638 .coefficient) true false)

def event70640 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13985⟩⟩, .operator (⟨3342, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70641RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70641RawTermsValid :
    exact70641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13985⟩⟩) exact70641RawTerms .large 70639 .exactZero (none)

def event70642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7176⟩⟩) 0 ⟨5533⟩ 65165

def event70643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7176⟩⟩) 1 ⟨6758⟩ 12024

def event70644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7176⟩⟩) (.product (.predecessor 0 70642 .coefficient) (.predecessor 1 70643 .coefficient) (⟨false, false, none, none, none⟩))

def event70645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7176⟩⟩, .operator (⟨65165, 0⟩, ⟨12024, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩)

def exact70646RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩]

theorem exact70646RawTermsValid :
    exact70646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7176⟩⟩) exact70646RawTerms .large 70644 .exactZero (none)

def event70647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13986⟩⟩) 0 ⟨7176⟩ 70646

def event70648 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13986⟩⟩) 1 ⟨13985⟩ 70641

def event70649 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13986⟩⟩) (.sum [.predecessor 0 70647 .coefficient, .predecessor 1 70648 .coefficient])

def exact70650RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70650RawTermsValid :
    exact70650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70650 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13986⟩⟩) exact70650RawTerms .large 70649 .exactZero (none)

def event70651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13987⟩⟩) 0 ⟨13986⟩ 70650

def event70652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13987⟩⟩) 1 ⟨72⟩ 12016

def event70653 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13987⟩⟩) (.sum [.predecessor 0 70651 .coefficient, .predecessor 1 70652 .coefficient])

def event70654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13987⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) [⟨.result 12016 .coefficient, false, none⟩])

def event70655 : Event := .survivorFold (1) 70654

def eventLeaf4400 : Array AnnotatedEvent := #[
  { event := event70400
    frameStart := 0 },
  { event := event70401
    frameStart := 0 },
  { event := event70402
    frameStart := 0 },
  { event := event70403
    frameStart := 0 },
  { event := event70404
    frameStart := 0 },
  { event := event70405
    frameStart := 0 },
  { event := event70406
    frameStart := 0 },
  { event := event70407
    frameStart := 0 },
  { event := event70408
    frameStart := 0 },
  { event := event70409
    frameStart := 0 },
  { event := event70410
    frameStart := 0 },
  { event := event70411
    frameStart := 0 },
  { event := event70412
    frameStart := 0 },
  { event := event70413
    frameStart := 0 },
  { event := event70414
    frameStart := 0 },
  { event := event70415
    frameStart := 0 }
]

def eventLeaf4401 : Array AnnotatedEvent := #[
  { event := event70416
    frameStart := 0 },
  { event := event70417
    frameStart := 70417 },
  { event := event70418
    frameStart := 70417 },
  { event := event70419
    frameStart := 70417 },
  { event := event70420
    frameStart := 70417 },
  { event := event70421
    frameStart := 70417 },
  { event := event70422
    frameStart := 70417 },
  { event := event70423
    frameStart := 70417 },
  { event := event70424
    frameStart := 70417 },
  { event := event70425
    frameStart := 70417 },
  { event := event70426
    frameStart := 70417 },
  { event := event70427
    frameStart := 70417 },
  { event := event70428
    frameStart := 70417 },
  { event := event70429
    frameStart := 70417 },
  { event := event70430
    frameStart := 70417 },
  { event := event70431
    frameStart := 70417 }
]

def eventLeaf4402 : Array AnnotatedEvent := #[
  { event := event70432
    frameStart := 70417 },
  { event := event70433
    frameStart := 70417 },
  { event := event70434
    frameStart := 70417 },
  { event := event70435
    frameStart := 70417 },
  { event := event70436
    frameStart := 70417 },
  { event := event70437
    frameStart := 70417 },
  { event := event70438
    frameStart := 70417 },
  { event := event70439
    frameStart := 70417 },
  { event := event70440
    frameStart := 70417 },
  { event := event70441
    frameStart := 70417 },
  { event := event70442
    frameStart := 70417 },
  { event := event70443
    frameStart := 70417 },
  { event := event70444
    frameStart := 70417 },
  { event := event70445
    frameStart := 70417 },
  { event := event70446
    frameStart := 70417 },
  { event := event70447
    frameStart := 70417 }
]

def eventLeaf4403 : Array AnnotatedEvent := #[
  { event := event70448
    frameStart := 70417 },
  { event := event70449
    frameStart := 70417 },
  { event := event70450
    frameStart := 70417 },
  { event := event70451
    frameStart := 70417 },
  { event := event70452
    frameStart := 70417 },
  { event := event70453
    frameStart := 70417 },
  { event := event70454
    frameStart := 70417 },
  { event := event70455
    frameStart := 70417 },
  { event := event70456
    frameStart := 70417 },
  { event := event70457
    frameStart := 70417 },
  { event := event70458
    frameStart := 70417 },
  { event := event70459
    frameStart := 70417 },
  { event := event70460
    frameStart := 70417 },
  { event := event70461
    frameStart := 70417 },
  { event := event70462
    frameStart := 70417 },
  { event := event70463
    frameStart := 70417 }
]

def eventLeaf4404 : Array AnnotatedEvent := #[
  { event := event70464
    frameStart := 70417 },
  { event := event70465
    frameStart := 70417 },
  { event := event70466
    frameStart := 70417 },
  { event := event70467
    frameStart := 70417 },
  { event := event70468
    frameStart := 70417 },
  { event := event70469
    frameStart := 70417 },
  { event := event70470
    frameStart := 70417 },
  { event := event70471
    frameStart := 70471 },
  { event := event70472
    frameStart := 70471 },
  { event := event70473
    frameStart := 70471 },
  { event := event70474
    frameStart := 70471 },
  { event := event70475
    frameStart := 70471 },
  { event := event70476
    frameStart := 70471 },
  { event := event70477
    frameStart := 70471 },
  { event := event70478
    frameStart := 70471 },
  { event := event70479
    frameStart := 70471 }
]

def eventLeaf4405 : Array AnnotatedEvent := #[
  { event := event70480
    frameStart := 70471 },
  { event := event70481
    frameStart := 70471 },
  { event := event70482
    frameStart := 70471 },
  { event := event70483
    frameStart := 70471 },
  { event := event70484
    frameStart := 70471 },
  { event := event70485
    frameStart := 70471 },
  { event := event70486
    frameStart := 70471 },
  { event := event70487
    frameStart := 70471 },
  { event := event70488
    frameStart := 70471 },
  { event := event70489
    frameStart := 70471 },
  { event := event70490
    frameStart := 70471 },
  { event := event70491
    frameStart := 70471 },
  { event := event70492
    frameStart := 70471 },
  { event := event70493
    frameStart := 70471 },
  { event := event70494
    frameStart := 70471 },
  { event := event70495
    frameStart := 70471 }
]

def eventLeaf4406 : Array AnnotatedEvent := #[
  { event := event70496
    frameStart := 70471 },
  { event := event70497
    frameStart := 70471 },
  { event := event70498
    frameStart := 70471 },
  { event := event70499
    frameStart := 70471 },
  { event := event70500
    frameStart := 70471 },
  { event := event70501
    frameStart := 70471 },
  { event := event70502
    frameStart := 70471 },
  { event := event70503
    frameStart := 70471 },
  { event := event70504
    frameStart := 70471 },
  { event := event70505
    frameStart := 70471 },
  { event := event70506
    frameStart := 70471 },
  { event := event70507
    frameStart := 70471 },
  { event := event70508
    frameStart := 70471 },
  { event := event70509
    frameStart := 70471 },
  { event := event70510
    frameStart := 70471 },
  { event := event70511
    frameStart := 70471 }
]

def eventLeaf4407 : Array AnnotatedEvent := #[
  { event := event70512
    frameStart := 70471 },
  { event := event70513
    frameStart := 70471 },
  { event := event70514
    frameStart := 70471 },
  { event := event70515
    frameStart := 70471 },
  { event := event70516
    frameStart := 70471 },
  { event := event70517
    frameStart := 70471 },
  { event := event70518
    frameStart := 70471 },
  { event := event70519
    frameStart := 70471 },
  { event := event70520
    frameStart := 70471 },
  { event := event70521
    frameStart := 70471 },
  { event := event70522
    frameStart := 70471 },
  { event := event70523
    frameStart := 70471 },
  { event := event70524
    frameStart := 70471 },
  { event := event70525
    frameStart := 70471 },
  { event := event70526
    frameStart := 70471 },
  { event := event70527
    frameStart := 70471 }
]

def eventLeaf4408 : Array AnnotatedEvent := #[
  { event := event70528
    frameStart := 70471 },
  { event := event70529
    frameStart := 70471 },
  { event := event70530
    frameStart := 70471 },
  { event := event70531
    frameStart := 70471 },
  { event := event70532
    frameStart := 70471 },
  { event := event70533
    frameStart := 70471 },
  { event := event70534
    frameStart := 70471 },
  { event := event70535
    frameStart := 70471 },
  { event := event70536
    frameStart := 70471 },
  { event := event70537
    frameStart := 70471 },
  { event := event70538
    frameStart := 70471 },
  { event := event70539
    frameStart := 70471 },
  { event := event70540
    frameStart := 70471 },
  { event := event70541
    frameStart := 70471 },
  { event := event70542
    frameStart := 70471 },
  { event := event70543
    frameStart := 70471 }
]

def eventLeaf4409 : Array AnnotatedEvent := #[
  { event := event70544
    frameStart := 70471 },
  { event := event70545
    frameStart := 70471 },
  { event := event70546
    frameStart := 70471 },
  { event := event70547
    frameStart := 70471 },
  { event := event70548
    frameStart := 70471 },
  { event := event70549
    frameStart := 70471 },
  { event := event70550
    frameStart := 70471 },
  { event := event70551
    frameStart := 70471 },
  { event := event70552
    frameStart := 70471 },
  { event := event70553
    frameStart := 70471 },
  { event := event70554
    frameStart := 70471 },
  { event := event70555
    frameStart := 70471 },
  { event := event70556
    frameStart := 70471 },
  { event := event70557
    frameStart := 70471 },
  { event := event70558
    frameStart := 70471 },
  { event := event70559
    frameStart := 70471 }
]

def eventLeaf4410 : Array AnnotatedEvent := #[
  { event := event70560
    frameStart := 70471 },
  { event := event70561
    frameStart := 70471 },
  { event := event70562
    frameStart := 70471 },
  { event := event70563
    frameStart := 70471 },
  { event := event70564
    frameStart := 70471 },
  { event := event70565
    frameStart := 70471 },
  { event := event70566
    frameStart := 70471 },
  { event := event70567
    frameStart := 70471 },
  { event := event70568
    frameStart := 70471 },
  { event := event70569
    frameStart := 70471 },
  { event := event70570
    frameStart := 70471 },
  { event := event70571
    frameStart := 70471 },
  { event := event70572
    frameStart := 70471 },
  { event := event70573
    frameStart := 70471 },
  { event := event70574
    frameStart := 70471 },
  { event := event70575
    frameStart := 0 }
]

def eventLeaf4411 : Array AnnotatedEvent := #[
  { event := event70576
    frameStart := 0 },
  { event := event70577
    frameStart := 0 },
  { event := event70578
    frameStart := 0 },
  { event := event70579
    frameStart := 0 },
  { event := event70580
    frameStart := 0 },
  { event := event70581
    frameStart := 0 },
  { event := event70582
    frameStart := 0 },
  { event := event70583
    frameStart := 0 },
  { event := event70584
    frameStart := 0 },
  { event := event70585
    frameStart := 0 },
  { event := event70586
    frameStart := 0 },
  { event := event70587
    frameStart := 0 },
  { event := event70588
    frameStart := 0 },
  { event := event70589
    frameStart := 0 },
  { event := event70590
    frameStart := 0 },
  { event := event70591
    frameStart := 0 }
]

def eventLeaf4412 : Array AnnotatedEvent := #[
  { event := event70592
    frameStart := 0 },
  { event := event70593
    frameStart := 0 },
  { event := event70594
    frameStart := 0 },
  { event := event70595
    frameStart := 0 },
  { event := event70596
    frameStart := 0 },
  { event := event70597
    frameStart := 0 },
  { event := event70598
    frameStart := 0 },
  { event := event70599
    frameStart := 0 },
  { event := event70600
    frameStart := 0 },
  { event := event70601
    frameStart := 0 },
  { event := event70602
    frameStart := 0 },
  { event := event70603
    frameStart := 0 },
  { event := event70604
    frameStart := 0 },
  { event := event70605
    frameStart := 0 },
  { event := event70606
    frameStart := 0 },
  { event := event70607
    frameStart := 0 }
]

def eventLeaf4413 : Array AnnotatedEvent := #[
  { event := event70608
    frameStart := 0 },
  { event := event70609
    frameStart := 0 },
  { event := event70610
    frameStart := 0 },
  { event := event70611
    frameStart := 0 },
  { event := event70612
    frameStart := 0 },
  { event := event70613
    frameStart := 0 },
  { event := event70614
    frameStart := 0 },
  { event := event70615
    frameStart := 0 },
  { event := event70616
    frameStart := 0 },
  { event := event70617
    frameStart := 0 },
  { event := event70618
    frameStart := 0 },
  { event := event70619
    frameStart := 0 },
  { event := event70620
    frameStart := 0 },
  { event := event70621
    frameStart := 0 },
  { event := event70622
    frameStart := 0 },
  { event := event70623
    frameStart := 0 }
]

def eventLeaf4414 : Array AnnotatedEvent := #[
  { event := event70624
    frameStart := 0 },
  { event := event70625
    frameStart := 0 },
  { event := event70626
    frameStart := 0 },
  { event := event70627
    frameStart := 0 },
  { event := event70628
    frameStart := 0 },
  { event := event70629
    frameStart := 0 },
  { event := event70630
    frameStart := 0 },
  { event := event70631
    frameStart := 0 },
  { event := event70632
    frameStart := 0 },
  { event := event70633
    frameStart := 0 },
  { event := event70634
    frameStart := 0 },
  { event := event70635
    frameStart := 0 },
  { event := event70636
    frameStart := 0 },
  { event := event70637
    frameStart := 0 },
  { event := event70638
    frameStart := 0 },
  { event := event70639
    frameStart := 0 }
]

def eventLeaf4415 : Array AnnotatedEvent := #[
  { event := event70640
    frameStart := 0 },
  { event := event70641
    frameStart := 0 },
  { event := event70642
    frameStart := 0 },
  { event := event70643
    frameStart := 0 },
  { event := event70644
    frameStart := 0 },
  { event := event70645
    frameStart := 0 },
  { event := event70646
    frameStart := 0 },
  { event := event70647
    frameStart := 0 },
  { event := event70648
    frameStart := 0 },
  { event := event70649
    frameStart := 0 },
  { event := event70650
    frameStart := 0 },
  { event := event70651
    frameStart := 0 },
  { event := event70652
    frameStart := 0 },
  { event := event70653
    frameStart := 0 },
  { event := event70654
    frameStart := 0 },
  { event := event70655
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events275
