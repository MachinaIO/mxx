import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events412

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event105472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24215⟩⟩) 0 ⟨6689⟩ 5477

def event105473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24215⟩⟩) 1 ⟨24214⟩ 98270

def event105474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24215⟩⟩) (.authority (.operator))

def exact105475RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩]

theorem exact105475RawTermsValid :
    exact105475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105475 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24215⟩⟩) exact105475RawTerms .large 105474 .exactZero (none)

def event105476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28041⟩⟩) 0 ⟨24215⟩ 105475

def event105477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28041⟩⟩) (.authority (.operator))

def exact105478RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩]

theorem exact105478RawTermsValid :
    exact105478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105478 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28041⟩⟩) exact105478RawTerms (.finite 8192) 105477 .exactZero (none)

def event105479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28043⟩⟩) 0 ⟨26132⟩ 98530

def event105480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28043⟩⟩) 1 ⟨28041⟩ 105478

def event105481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28043⟩⟩) (.product (.predecessor 0 105479 .coefficient) (.predecessor 1 105480 .coefficient) (⟨false, false, none, none, none⟩))

def event105482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28043⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) [⟨.result 105478 .coefficient, false, none⟩])

def event105483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28043⟩⟩) (.product (.result 98530 .summary) (.transfer 105482) (⟨false, false, none, none, none⟩))

def event105484 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28043⟩⟩, .operator (⟨98530, 0⟩, ⟨105478, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩)

def event105485 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28043⟩⟩, .operator (⟨98530, 1⟩, ⟨105478, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩)

def event105486 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28043⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28041⟩⟩) ⟨24215⟩ 105475)

def event105487 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28043⟩⟩, .relation 105486 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (-1)⟩)

def exact105488RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (-1)⟩]

theorem exact105488RawTermsValid :
    exact105488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28043⟩⟩) exact105488RawTerms .large 105481 (.finite 1292113297018323992576) (some (105483))

def event105489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21461⟩⟩) 0 ⟨16050⟩ 4790

def event105490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21461⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact105491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩]

theorem exact105491RawTermsValid :
    exact105491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21461⟩⟩) exact105491RawTerms (.finite 136065468) 105490 .exactZero (none)

def event105492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21463⟩⟩) 0 ⟨21461⟩ 105491

def event105493 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21463⟩⟩) 1 ⟨2348⟩ 4

def event105494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21463⟩⟩) (.scale (.predecessor 0 105492 .coefficient) (.value (.predecessor 1 105493 .coefficient)))

def exact105495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩]

theorem exact105495RawTermsValid :
    exact105495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105495 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21463⟩⟩) exact105495RawTerms (.finite 136065468) 105494 .exactZero (none)

def event105496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21464⟩⟩) 0 ⟨5509⟩ 94462

def event105497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21464⟩⟩) 1 ⟨21463⟩ 105495

def event105498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21464⟩⟩) (.product (.predecessor 0 105496 .coefficient) (.predecessor 1 105497 .coefficient) (⟨false, false, none, none, none⟩))

def event105499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21464⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) [⟨.result 105491 .coefficient, false, none⟩])

def event105500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21464⟩⟩) (.product (.result 94462 .summary) (.transfer 105499) (⟨false, false, none, none, none⟩))

def event105501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21464⟩⟩, .operator (⟨94462, 0⟩, ⟨105495, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩)

def event105502 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21462⟩⟩)

def event105503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event105504 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event105505 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event105506 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event105507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 105506

def event105508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 105504

def event105509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 105507 .coefficient) (.value (.predecessor 1 105508 .coefficient)))

def event105510 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event105511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11541⟩⟩) 0 ⟨5503⟩ 105510

def event105512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11541⟩⟩) (.authority (.programFamilyFact))

def exact105513RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩], []⟩, (1)⟩]

theorem exact105513RawTermsValid :
    exact105513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105513 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11541⟩⟩) exact105513RawTerms (.finite 22) 105512 .exactZero (none)

def event105514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14397⟩⟩) 0 ⟨5503⟩ 105510

def event105515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14397⟩⟩) (.authority (.programFamilyFact))

def exact105516RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩, (1)⟩]

theorem exact105516RawTermsValid :
    exact105516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105516 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14397⟩⟩) exact105516RawTerms (.finite 22) 105515 .exactZero (none)

def event105517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14398⟩⟩) 0 ⟨14397⟩ 105516

def event105518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14398⟩⟩) 1 ⟨11541⟩ 105513

def event105519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14398⟩⟩) (.product (.predecessor 0 105517 .coefficient) (.predecessor 1 105518 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14398⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩) [⟨.result 105516 .coefficient, true, some 1⟩, ⟨.result 105513 .coefficient, true, some 1⟩])

def event105521 : Event := .survivorFold (1) 105520

def exact105522RawTerms : List Term := []

theorem exact105522RawTermsValid :
    exact105522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14398⟩⟩) exact105522RawTerms (.finite 484) 105519 (.finite 484) (some (105520))

def event105523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14399⟩⟩) 0 ⟨14398⟩ 105522

def event105524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14399⟩⟩) (.identity (.predecessor 0 105523 .coefficient))

def event105525 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14399⟩⟩) (.finite 484)

def event105526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16049⟩⟩) 0 ⟨14399⟩ 105525

def event105527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16049⟩⟩) (.authority (.programFamilyFact))

def exact105528RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], []⟩, (1)⟩]

theorem exact105528RawTermsValid :
    exact105528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105528 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16049⟩⟩) exact105528RawTerms (.finite 22) 105527 .exactZero (none)

def event105529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16050⟩⟩) 0 ⟨16049⟩ 105528

def event105530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16050⟩⟩) (.identity (.predecessor 0 105529 .coefficient))

def event105531 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16050⟩⟩) (.finite 22)

def event105532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21461⟩⟩) 0 ⟨16050⟩ 105531

def event105533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21461⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact105534RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩]

theorem exact105534RawTermsValid :
    exact105534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21461⟩⟩) exact105534RawTerms (.finite 136065468) 105533 .exactZero (none)

def event105535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact105536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact105536RawTermsValid :
    exact105536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105536 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact105536RawTerms .large 105535 .exactZero (none)

def event105537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21462⟩⟩) 0 ⟨6⟩ 105536

def event105538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21462⟩⟩) 1 ⟨21461⟩ 105534

def event105539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21462⟩⟩) (.product (.predecessor 0 105537 .coefficient) (.predecessor 1 105538 .coefficient) (⟨false, false, none, none, none⟩))

def event105540 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21462⟩⟩, .operator (⟨105536, 0⟩, ⟨105534, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩)

def exact105541RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩]

theorem exact105541RawTermsValid :
    exact105541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105541 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21462⟩⟩) exact105541RawTerms .large 105539 .exactZero (none)

def event105542 : Event := .preFoldPolynomial 105541 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩] .exactZero none

def exact105543RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩, (1)⟩]

def event105543 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21462⟩⟩) 105542 exact105543RawTerms .large 105539 .exactZero (none)

def event105544 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28047⟩⟩)

def event105545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event105546 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event105547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event105548 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event105549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 105548

def event105550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 105546

def event105551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 105549 .coefficient) (.value (.predecessor 1 105550 .coefficient)))

def event105552 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event105553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11541⟩⟩) 0 ⟨5503⟩ 105552

def event105554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11541⟩⟩) (.authority (.programFamilyFact))

def exact105555RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩], []⟩, (1)⟩]

theorem exact105555RawTermsValid :
    exact105555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105555 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11541⟩⟩) exact105555RawTerms (.finite 22) 105554 .exactZero (none)

def event105556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14397⟩⟩) 0 ⟨5503⟩ 105552

def event105557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14397⟩⟩) (.authority (.programFamilyFact))

def exact105558RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩, (1)⟩]

theorem exact105558RawTermsValid :
    exact105558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105558 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14397⟩⟩) exact105558RawTerms (.finite 22) 105557 .exactZero (none)

def event105559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14398⟩⟩) 0 ⟨14397⟩ 105558

def event105560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14398⟩⟩) 1 ⟨11541⟩ 105555

def event105561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14398⟩⟩) (.product (.predecessor 0 105559 .coefficient) (.predecessor 1 105560 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105562 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14398⟩⟩, .operator (⟨105558, 0⟩, ⟨105555, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩, (1)⟩)

def exact105563RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11541⟩⟩, ⟨.program ⟨214⟩, ⟨14397⟩⟩], []⟩, (1)⟩]

theorem exact105563RawTermsValid :
    exact105563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105563 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14398⟩⟩) exact105563RawTerms (.finite 484) 105561 .exactZero (none)

def event105564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14399⟩⟩) 0 ⟨14398⟩ 105563

def event105565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14399⟩⟩) (.identity (.predecessor 0 105564 .coefficient))

def event105566 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14399⟩⟩) (.finite 484)

def event105567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16049⟩⟩) 0 ⟨14399⟩ 105566

def event105568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16049⟩⟩) (.authority (.programFamilyFact))

def exact105569RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], []⟩, (1)⟩]

theorem exact105569RawTermsValid :
    exact105569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16049⟩⟩) exact105569RawTerms (.finite 22) 105568 .exactZero (none)

def event105570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16050⟩⟩) 0 ⟨16049⟩ 105569

def event105571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16050⟩⟩) (.identity (.predecessor 0 105570 .coefficient))

def event105572 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16050⟩⟩) (.finite 22)

def event105573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24214⟩⟩) 0 ⟨16050⟩ 105572

def event105574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24214⟩⟩) (.authority (.programFamilyFact))

def event105575 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24214⟩⟩) (.finite 3720)

def event105576 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event105577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24215⟩⟩) 0 ⟨6689⟩ 105576

def event105578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24215⟩⟩) 1 ⟨24214⟩ 105575

def event105579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24215⟩⟩) (.authority (.operator))

def exact105580RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩]

theorem exact105580RawTermsValid :
    exact105580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24215⟩⟩) exact105580RawTerms .large 105579 .exactZero (none)

def event105581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28041⟩⟩) 0 ⟨24215⟩ 105580

def event105582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28041⟩⟩) (.authority (.operator))

def exact105583RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩]

theorem exact105583RawTermsValid :
    exact105583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105583 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28041⟩⟩) exact105583RawTerms (.finite 8192) 105582 .exactZero (none)

def event105584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event105585 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event105586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16126⟩⟩) 0 ⟨16050⟩ 105572

def event105587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16126⟩⟩) 1 ⟨110⟩ 105585

def event105588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16126⟩⟩) (.sum [.predecessor 0 105586 .coefficient, .predecessor 1 105587 .coefficient])

def event105589 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16126⟩⟩) (.finite 22)

def event105590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16127⟩⟩) 0 ⟨16126⟩ 105589

def event105591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16127⟩⟩) (.identity (.predecessor 0 105590 .coefficient))

def exact105592RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], []⟩, (1)⟩]

theorem exact105592RawTermsValid :
    exact105592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16127⟩⟩) exact105592RawTerms (.finite 22) 105591 .exactZero (none)

def event105593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact105594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105594RawTermsValid :
    exact105594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105594 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact105594RawTerms .large 105593 .exactZero (none)

def event105595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16128⟩⟩) 0 ⟨6544⟩ 105594

def event105596 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16128⟩⟩) 1 ⟨16127⟩ 105592

def event105597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16128⟩⟩) (.product (.predecessor 0 105595 .coefficient) (.predecessor 1 105596 .coefficient) (⟨false, false, none, none, none⟩))

def event105598 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16128⟩⟩, .operator (⟨105594, 0⟩, ⟨105592, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105599RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105599RawTermsValid :
    exact105599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105599 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16128⟩⟩) exact105599RawTerms .large 105597 .exactZero (none)

def event105600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 105576

def event105601 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact105602RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact105602RawTermsValid :
    exact105602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105602 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact105602RawTerms .large 105601 .exactZero (none)

def event105603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16129⟩⟩) 0 ⟨6698⟩ 105602

def event105604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16129⟩⟩) 1 ⟨16128⟩ 105599

def event105605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16129⟩⟩) (.sum [.predecessor 0 105603 .coefficient, .predecessor 1 105604 .coefficient])

def exact105606RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105606RawTermsValid :
    exact105606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105606 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16129⟩⟩) exact105606RawTerms .large 105605 .exactZero (none)

def event105607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28042⟩⟩) 0 ⟨16129⟩ 105606

def event105608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28042⟩⟩) 1 ⟨28041⟩ 105583

def event105609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28042⟩⟩) (.product (.predecessor 0 105607 .coefficient) (.predecessor 1 105608 .coefficient) (⟨false, false, none, none, none⟩))

def event105610 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28042⟩⟩, .operator (⟨105606, 0⟩, ⟨105583, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩)

def event105611 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28042⟩⟩, .operator (⟨105606, 1⟩, ⟨105583, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩)

def event105612 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28042⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28041⟩⟩) ⟨24215⟩ 105580)

def event105613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28042⟩⟩, .relation 105612 0, ⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (-1)⟩)

def exact105614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (-1)⟩]

theorem exact105614RawTermsValid :
    exact105614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28042⟩⟩) exact105614RawTerms .large 105609 .exactZero (none)

def event105615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18016⟩⟩) 0 ⟨16050⟩ 105572

def event105616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18016⟩⟩) (.authority (.programFamilyFact))

def exact105617RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], []⟩, (1)⟩]

theorem exact105617RawTermsValid :
    exact105617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18016⟩⟩) exact105617RawTerms (.finite 22) 105616 .exactZero (none)

def event105618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18021⟩⟩) 0 ⟨6544⟩ 105594

def event105619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18021⟩⟩) 1 ⟨18016⟩ 105617

def event105620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18021⟩⟩) (.product (.predecessor 0 105618 .coefficient) (.predecessor 1 105619 .coefficient) (⟨false, true, none, none, some 1⟩))

def event105621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18021⟩⟩, .operator (⟨105594, 0⟩, ⟨105617, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact105622RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact105622RawTermsValid :
    exact105622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105622 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18021⟩⟩) exact105622RawTerms .large 105620 .exactZero (none)

def event105623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6724⟩⟩) 0 ⟨6689⟩ 105576

def event105624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6724⟩⟩) (.authority (.operator))

def exact105625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩]

theorem exact105625RawTermsValid :
    exact105625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105625 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6724⟩⟩) exact105625RawTerms .large 105624 .exactZero (none)

def event105626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18022⟩⟩) 0 ⟨6724⟩ 105625

def event105627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18022⟩⟩) 1 ⟨18021⟩ 105622

def event105628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18022⟩⟩) (.sum [.predecessor 0 105626 .coefficient, .predecessor 1 105627 .coefficient])

def exact105629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105629RawTermsValid :
    exact105629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18022⟩⟩) exact105629RawTerms .large 105628 .exactZero (none)

def event105630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28047⟩⟩) 0 ⟨18022⟩ 105629

def event105631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28047⟩⟩) 1 ⟨28042⟩ 105614

def event105632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28047⟩⟩) (.sum [.predecessor 0 105630 .coefficient, .predecessor 1 105631 .coefficient])

def exact105633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105633RawTermsValid :
    exact105633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28047⟩⟩) exact105633RawTerms .large 105632 .exactZero (none)

def event105634 : Event := .preFoldPolynomial 105633 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact105635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event105635 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28047⟩⟩) 105634 exact105635RawTerms .large 105632 .exactZero (none)

def event105636 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16050⟩⟩) ⟨⟨137⟩, ⟨45⟩, ⟨109⟩⟩ ⟨105502, 105636⟩

def event105637 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21464⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (1) 0 2 (.universal 105636 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21461⟩⟩]⟩) (none) 105635)

def event105638 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21464⟩⟩, .relation 105637 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩)

def event105639 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21464⟩⟩, .relation 105637 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩)

def event105640 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21464⟩⟩, .relation 105637 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩)

def event105641 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21464⟩⟩, .relation 105637 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105642RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105642RawTermsValid :
    exact105642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105642 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21464⟩⟩) exact105642RawTerms .large 105498 (.finite 1811303510016) (some (105500))

def event105643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28044⟩⟩) 0 ⟨21464⟩ 105642

def event105644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28044⟩⟩) 1 ⟨28043⟩ 105488

def event105645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28044⟩⟩) (.sum [.predecessor 0 105643 .coefficient, .predecessor 1 105644 .coefficient])

def event105646 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28044⟩⟩, .operator (⟨105642, 0⟩, ⟨105488, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩, (1)⟩)

def event105647 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28044⟩⟩, .operator (⟨105642, 2⟩, ⟨105488, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16049⟩⟩], [⟨.program ⟨214⟩, ⟨24215⟩⟩]⟩, (-1)⟩)

def event105648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28044⟩⟩) (.sum [.result 105642 .summary, .result 105488 .summary])

def exact105649RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105649RawTermsValid :
    exact105649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105649 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28044⟩⟩) exact105649RawTerms .large 105645 (.finite 1292113298829627502592) (some (105648))

def event105650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28045⟩⟩) 0 ⟨28044⟩ 105649

def event105651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28045⟩⟩) 1 ⟨6638⟩ 5699

def event105652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28045⟩⟩) (.product (.predecessor 0 105650 .coefficient) (.predecessor 1 105651 .coefficient) (⟨false, false, none, none, none⟩))

def event105653 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28045⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) [⟨.result 5695 .coefficient, false, none⟩])

def event105654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28045⟩⟩) (.product (.result 105649 .summary) (.transfer 105653) (⟨false, false, none, none, none⟩))

def event105655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28045⟩⟩, .operator (⟨105649, 0⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩)

def event105656 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28045⟩⟩, .operator (⟨105649, 1⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (-1)⟩)

def event105657 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28045⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6637⟩⟩) ⟨6590⟩ 5692)

def event105658 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28045⟩⟩, .relation 105657 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact105659RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact105659RawTermsValid :
    exact105659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105659 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28045⟩⟩) exact105659RawTerms .large 105652 (.finite 4742076480517514208552681472) (some (105654))

def event105660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24152⟩⟩) 0 ⟨6689⟩ 5477

def event105661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24152⟩⟩) 1 ⟨24151⟩ 98704

def event105662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24152⟩⟩) (.authority (.operator))

def exact105663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24152⟩⟩]⟩, (1)⟩]

theorem exact105663RawTermsValid :
    exact105663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24152⟩⟩) exact105663RawTerms .large 105662 .exactZero (none)

def event105664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27824⟩⟩) 0 ⟨24152⟩ 105663

def event105665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27824⟩⟩) (.authority (.operator))

def exact105666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩, (1)⟩]

theorem exact105666RawTermsValid :
    exact105666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27824⟩⟩) exact105666RawTerms (.finite 8192) 105665 .exactZero (none)

def event105667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27826⟩⟩) 0 ⟨26055⟩ 98964

def event105668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27826⟩⟩) 1 ⟨27824⟩ 105666

def event105669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27826⟩⟩) (.product (.predecessor 0 105667 .coefficient) (.predecessor 1 105668 .coefficient) (⟨false, false, none, none, none⟩))

def event105670 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27826⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩) [⟨.result 105666 .coefficient, false, none⟩])

def event105671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27826⟩⟩) (.product (.result 98964 .summary) (.transfer 105670) (⟨false, false, none, none, none⟩))

def event105672 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27826⟩⟩, .operator (⟨98964, 0⟩, ⟨105666, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩, (1)⟩)

def event105673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27826⟩⟩, .operator (⟨98964, 1⟩, ⟨105666, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩, (-1)⟩)

def event105674 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27826⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27824⟩⟩) ⟨24152⟩ 105663)

def event105675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27826⟩⟩, .relation 105674 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨24152⟩⟩]⟩, (-1)⟩)

def exact105676RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27824⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15930⟩⟩], [⟨.program ⟨214⟩, ⟨24152⟩⟩]⟩, (-1)⟩]

theorem exact105676RawTermsValid :
    exact105676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105676 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27826⟩⟩) exact105676RawTerms .large 105669 (.finite 1292068472128282820608) (some (105671))

def event105677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21317⟩⟩) 0 ⟨15931⟩ 4813

def event105678 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21317⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact105679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21317⟩⟩]⟩, (1)⟩]

theorem exact105679RawTermsValid :
    exact105679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105679 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21317⟩⟩) exact105679RawTerms (.finite 136065468) 105678 .exactZero (none)

def event105680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21319⟩⟩) 0 ⟨21317⟩ 105679

def event105681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21319⟩⟩) 1 ⟨2348⟩ 4

def event105682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21319⟩⟩) (.scale (.predecessor 0 105680 .coefficient) (.value (.predecessor 1 105681 .coefficient)))

def exact105683RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21317⟩⟩]⟩, (1)⟩]

theorem exact105683RawTermsValid :
    exact105683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105683 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21319⟩⟩) exact105683RawTerms (.finite 136065468) 105682 .exactZero (none)

def event105684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21320⟩⟩) 0 ⟨5509⟩ 94462

def event105685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21320⟩⟩) 1 ⟨21319⟩ 105683

def event105686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21320⟩⟩) (.product (.predecessor 0 105684 .coefficient) (.predecessor 1 105685 .coefficient) (⟨false, false, none, none, none⟩))

def event105687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21320⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21317⟩⟩]⟩) [⟨.result 105679 .coefficient, false, none⟩])

def event105688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21320⟩⟩) (.product (.result 94462 .summary) (.transfer 105687) (⟨false, false, none, none, none⟩))

def event105689 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21320⟩⟩, .operator (⟨94462, 0⟩, ⟨105683, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21317⟩⟩]⟩, (1)⟩)

def event105690 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21318⟩⟩)

def event105691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event105692 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event105693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event105694 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event105695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 105694

def event105696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 105692

def event105697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 105695 .coefficient) (.value (.predecessor 1 105696 .coefficient)))

def event105698 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event105699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11457⟩⟩) 0 ⟨5503⟩ 105698

def event105700 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11457⟩⟩) (.authority (.programFamilyFact))

def exact105701RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11457⟩⟩], []⟩, (1)⟩]

theorem exact105701RawTermsValid :
    exact105701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105701 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11457⟩⟩) exact105701RawTerms (.finite 18) 105700 .exactZero (none)

def event105702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14180⟩⟩) 0 ⟨5503⟩ 105698

def event105703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14180⟩⟩) (.authority (.programFamilyFact))

def exact105704RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14180⟩⟩], []⟩, (1)⟩]

theorem exact105704RawTermsValid :
    exact105704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105704 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14180⟩⟩) exact105704RawTerms (.finite 18) 105703 .exactZero (none)

def event105705 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14181⟩⟩) 0 ⟨14180⟩ 105704

def event105706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14181⟩⟩) 1 ⟨11457⟩ 105701

def event105707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14181⟩⟩) (.product (.predecessor 0 105705 .coefficient) (.predecessor 1 105706 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14181⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11457⟩⟩, ⟨.program ⟨214⟩, ⟨14180⟩⟩], []⟩) [⟨.result 105704 .coefficient, true, some 1⟩, ⟨.result 105701 .coefficient, true, some 1⟩])

def event105709 : Event := .survivorFold (1) 105708

def exact105710RawTerms : List Term := []

theorem exact105710RawTermsValid :
    exact105710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105710 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14181⟩⟩) exact105710RawTerms (.finite 324) 105707 (.finite 324) (some (105708))

def event105711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14182⟩⟩) 0 ⟨14181⟩ 105710

def event105712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14182⟩⟩) (.identity (.predecessor 0 105711 .coefficient))

def event105713 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14182⟩⟩) (.finite 324)

def event105714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15930⟩⟩) 0 ⟨14182⟩ 105713

def event105715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15930⟩⟩) (.authority (.programFamilyFact))

def exact105716RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15930⟩⟩], []⟩, (1)⟩]

theorem exact105716RawTermsValid :
    exact105716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15930⟩⟩) exact105716RawTerms (.finite 18) 105715 .exactZero (none)

def event105717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15931⟩⟩) 0 ⟨15930⟩ 105716

def event105718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15931⟩⟩) (.identity (.predecessor 0 105717 .coefficient))

def event105719 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15931⟩⟩) (.finite 18)

def event105720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21317⟩⟩) 0 ⟨15931⟩ 105719

def event105721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21317⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact105722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21317⟩⟩]⟩, (1)⟩]

theorem exact105722RawTermsValid :
    exact105722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105722 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21317⟩⟩) exact105722RawTerms (.finite 136065468) 105721 .exactZero (none)

def event105723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact105724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact105724RawTermsValid :
    exact105724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105724 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact105724RawTerms .large 105723 .exactZero (none)

def event105725 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21318⟩⟩) 0 ⟨6⟩ 105724

def event105726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21318⟩⟩) 1 ⟨21317⟩ 105722

def event105727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21318⟩⟩) (.product (.predecessor 0 105725 .coefficient) (.predecessor 1 105726 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf6592 : Array AnnotatedEvent := #[
  { event := event105472
    frameStart := 0 },
  { event := event105473
    frameStart := 0 },
  { event := event105474
    frameStart := 0 },
  { event := event105475
    frameStart := 0 },
  { event := event105476
    frameStart := 0 },
  { event := event105477
    frameStart := 0 },
  { event := event105478
    frameStart := 0 },
  { event := event105479
    frameStart := 0 },
  { event := event105480
    frameStart := 0 },
  { event := event105481
    frameStart := 0 },
  { event := event105482
    frameStart := 0 },
  { event := event105483
    frameStart := 0 },
  { event := event105484
    frameStart := 0 },
  { event := event105485
    frameStart := 0 },
  { event := event105486
    frameStart := 0 },
  { event := event105487
    frameStart := 0 }
]

def eventLeaf6593 : Array AnnotatedEvent := #[
  { event := event105488
    frameStart := 0 },
  { event := event105489
    frameStart := 0 },
  { event := event105490
    frameStart := 0 },
  { event := event105491
    frameStart := 0 },
  { event := event105492
    frameStart := 0 },
  { event := event105493
    frameStart := 0 },
  { event := event105494
    frameStart := 0 },
  { event := event105495
    frameStart := 0 },
  { event := event105496
    frameStart := 0 },
  { event := event105497
    frameStart := 0 },
  { event := event105498
    frameStart := 0 },
  { event := event105499
    frameStart := 0 },
  { event := event105500
    frameStart := 0 },
  { event := event105501
    frameStart := 0 },
  { event := event105502
    frameStart := 105502 },
  { event := event105503
    frameStart := 105502 }
]

def eventLeaf6594 : Array AnnotatedEvent := #[
  { event := event105504
    frameStart := 105502 },
  { event := event105505
    frameStart := 105502 },
  { event := event105506
    frameStart := 105502 },
  { event := event105507
    frameStart := 105502 },
  { event := event105508
    frameStart := 105502 },
  { event := event105509
    frameStart := 105502 },
  { event := event105510
    frameStart := 105502 },
  { event := event105511
    frameStart := 105502 },
  { event := event105512
    frameStart := 105502 },
  { event := event105513
    frameStart := 105502 },
  { event := event105514
    frameStart := 105502 },
  { event := event105515
    frameStart := 105502 },
  { event := event105516
    frameStart := 105502 },
  { event := event105517
    frameStart := 105502 },
  { event := event105518
    frameStart := 105502 },
  { event := event105519
    frameStart := 105502 }
]

def eventLeaf6595 : Array AnnotatedEvent := #[
  { event := event105520
    frameStart := 105502 },
  { event := event105521
    frameStart := 105502 },
  { event := event105522
    frameStart := 105502 },
  { event := event105523
    frameStart := 105502 },
  { event := event105524
    frameStart := 105502 },
  { event := event105525
    frameStart := 105502 },
  { event := event105526
    frameStart := 105502 },
  { event := event105527
    frameStart := 105502 },
  { event := event105528
    frameStart := 105502 },
  { event := event105529
    frameStart := 105502 },
  { event := event105530
    frameStart := 105502 },
  { event := event105531
    frameStart := 105502 },
  { event := event105532
    frameStart := 105502 },
  { event := event105533
    frameStart := 105502 },
  { event := event105534
    frameStart := 105502 },
  { event := event105535
    frameStart := 105502 }
]

def eventLeaf6596 : Array AnnotatedEvent := #[
  { event := event105536
    frameStart := 105502 },
  { event := event105537
    frameStart := 105502 },
  { event := event105538
    frameStart := 105502 },
  { event := event105539
    frameStart := 105502 },
  { event := event105540
    frameStart := 105502 },
  { event := event105541
    frameStart := 105502 },
  { event := event105542
    frameStart := 105502 },
  { event := event105543
    frameStart := 105502 },
  { event := event105544
    frameStart := 105544 },
  { event := event105545
    frameStart := 105544 },
  { event := event105546
    frameStart := 105544 },
  { event := event105547
    frameStart := 105544 },
  { event := event105548
    frameStart := 105544 },
  { event := event105549
    frameStart := 105544 },
  { event := event105550
    frameStart := 105544 },
  { event := event105551
    frameStart := 105544 }
]

def eventLeaf6597 : Array AnnotatedEvent := #[
  { event := event105552
    frameStart := 105544 },
  { event := event105553
    frameStart := 105544 },
  { event := event105554
    frameStart := 105544 },
  { event := event105555
    frameStart := 105544 },
  { event := event105556
    frameStart := 105544 },
  { event := event105557
    frameStart := 105544 },
  { event := event105558
    frameStart := 105544 },
  { event := event105559
    frameStart := 105544 },
  { event := event105560
    frameStart := 105544 },
  { event := event105561
    frameStart := 105544 },
  { event := event105562
    frameStart := 105544 },
  { event := event105563
    frameStart := 105544 },
  { event := event105564
    frameStart := 105544 },
  { event := event105565
    frameStart := 105544 },
  { event := event105566
    frameStart := 105544 },
  { event := event105567
    frameStart := 105544 }
]

def eventLeaf6598 : Array AnnotatedEvent := #[
  { event := event105568
    frameStart := 105544 },
  { event := event105569
    frameStart := 105544 },
  { event := event105570
    frameStart := 105544 },
  { event := event105571
    frameStart := 105544 },
  { event := event105572
    frameStart := 105544 },
  { event := event105573
    frameStart := 105544 },
  { event := event105574
    frameStart := 105544 },
  { event := event105575
    frameStart := 105544 },
  { event := event105576
    frameStart := 105544 },
  { event := event105577
    frameStart := 105544 },
  { event := event105578
    frameStart := 105544 },
  { event := event105579
    frameStart := 105544 },
  { event := event105580
    frameStart := 105544 },
  { event := event105581
    frameStart := 105544 },
  { event := event105582
    frameStart := 105544 },
  { event := event105583
    frameStart := 105544 }
]

def eventLeaf6599 : Array AnnotatedEvent := #[
  { event := event105584
    frameStart := 105544 },
  { event := event105585
    frameStart := 105544 },
  { event := event105586
    frameStart := 105544 },
  { event := event105587
    frameStart := 105544 },
  { event := event105588
    frameStart := 105544 },
  { event := event105589
    frameStart := 105544 },
  { event := event105590
    frameStart := 105544 },
  { event := event105591
    frameStart := 105544 },
  { event := event105592
    frameStart := 105544 },
  { event := event105593
    frameStart := 105544 },
  { event := event105594
    frameStart := 105544 },
  { event := event105595
    frameStart := 105544 },
  { event := event105596
    frameStart := 105544 },
  { event := event105597
    frameStart := 105544 },
  { event := event105598
    frameStart := 105544 },
  { event := event105599
    frameStart := 105544 }
]

def eventLeaf6600 : Array AnnotatedEvent := #[
  { event := event105600
    frameStart := 105544 },
  { event := event105601
    frameStart := 105544 },
  { event := event105602
    frameStart := 105544 },
  { event := event105603
    frameStart := 105544 },
  { event := event105604
    frameStart := 105544 },
  { event := event105605
    frameStart := 105544 },
  { event := event105606
    frameStart := 105544 },
  { event := event105607
    frameStart := 105544 },
  { event := event105608
    frameStart := 105544 },
  { event := event105609
    frameStart := 105544 },
  { event := event105610
    frameStart := 105544 },
  { event := event105611
    frameStart := 105544 },
  { event := event105612
    frameStart := 105544 },
  { event := event105613
    frameStart := 105544 },
  { event := event105614
    frameStart := 105544 },
  { event := event105615
    frameStart := 105544 }
]

def eventLeaf6601 : Array AnnotatedEvent := #[
  { event := event105616
    frameStart := 105544 },
  { event := event105617
    frameStart := 105544 },
  { event := event105618
    frameStart := 105544 },
  { event := event105619
    frameStart := 105544 },
  { event := event105620
    frameStart := 105544 },
  { event := event105621
    frameStart := 105544 },
  { event := event105622
    frameStart := 105544 },
  { event := event105623
    frameStart := 105544 },
  { event := event105624
    frameStart := 105544 },
  { event := event105625
    frameStart := 105544 },
  { event := event105626
    frameStart := 105544 },
  { event := event105627
    frameStart := 105544 },
  { event := event105628
    frameStart := 105544 },
  { event := event105629
    frameStart := 105544 },
  { event := event105630
    frameStart := 105544 },
  { event := event105631
    frameStart := 105544 }
]

def eventLeaf6602 : Array AnnotatedEvent := #[
  { event := event105632
    frameStart := 105544 },
  { event := event105633
    frameStart := 105544 },
  { event := event105634
    frameStart := 105544 },
  { event := event105635
    frameStart := 105544 },
  { event := event105636
    frameStart := 0 },
  { event := event105637
    frameStart := 0 },
  { event := event105638
    frameStart := 0 },
  { event := event105639
    frameStart := 0 },
  { event := event105640
    frameStart := 0 },
  { event := event105641
    frameStart := 0 },
  { event := event105642
    frameStart := 0 },
  { event := event105643
    frameStart := 0 },
  { event := event105644
    frameStart := 0 },
  { event := event105645
    frameStart := 0 },
  { event := event105646
    frameStart := 0 },
  { event := event105647
    frameStart := 0 }
]

def eventLeaf6603 : Array AnnotatedEvent := #[
  { event := event105648
    frameStart := 0 },
  { event := event105649
    frameStart := 0 },
  { event := event105650
    frameStart := 0 },
  { event := event105651
    frameStart := 0 },
  { event := event105652
    frameStart := 0 },
  { event := event105653
    frameStart := 0 },
  { event := event105654
    frameStart := 0 },
  { event := event105655
    frameStart := 0 },
  { event := event105656
    frameStart := 0 },
  { event := event105657
    frameStart := 0 },
  { event := event105658
    frameStart := 0 },
  { event := event105659
    frameStart := 0 },
  { event := event105660
    frameStart := 0 },
  { event := event105661
    frameStart := 0 },
  { event := event105662
    frameStart := 0 },
  { event := event105663
    frameStart := 0 }
]

def eventLeaf6604 : Array AnnotatedEvent := #[
  { event := event105664
    frameStart := 0 },
  { event := event105665
    frameStart := 0 },
  { event := event105666
    frameStart := 0 },
  { event := event105667
    frameStart := 0 },
  { event := event105668
    frameStart := 0 },
  { event := event105669
    frameStart := 0 },
  { event := event105670
    frameStart := 0 },
  { event := event105671
    frameStart := 0 },
  { event := event105672
    frameStart := 0 },
  { event := event105673
    frameStart := 0 },
  { event := event105674
    frameStart := 0 },
  { event := event105675
    frameStart := 0 },
  { event := event105676
    frameStart := 0 },
  { event := event105677
    frameStart := 0 },
  { event := event105678
    frameStart := 0 },
  { event := event105679
    frameStart := 0 }
]

def eventLeaf6605 : Array AnnotatedEvent := #[
  { event := event105680
    frameStart := 0 },
  { event := event105681
    frameStart := 0 },
  { event := event105682
    frameStart := 0 },
  { event := event105683
    frameStart := 0 },
  { event := event105684
    frameStart := 0 },
  { event := event105685
    frameStart := 0 },
  { event := event105686
    frameStart := 0 },
  { event := event105687
    frameStart := 0 },
  { event := event105688
    frameStart := 0 },
  { event := event105689
    frameStart := 0 },
  { event := event105690
    frameStart := 105690 },
  { event := event105691
    frameStart := 105690 },
  { event := event105692
    frameStart := 105690 },
  { event := event105693
    frameStart := 105690 },
  { event := event105694
    frameStart := 105690 },
  { event := event105695
    frameStart := 105690 }
]

def eventLeaf6606 : Array AnnotatedEvent := #[
  { event := event105696
    frameStart := 105690 },
  { event := event105697
    frameStart := 105690 },
  { event := event105698
    frameStart := 105690 },
  { event := event105699
    frameStart := 105690 },
  { event := event105700
    frameStart := 105690 },
  { event := event105701
    frameStart := 105690 },
  { event := event105702
    frameStart := 105690 },
  { event := event105703
    frameStart := 105690 },
  { event := event105704
    frameStart := 105690 },
  { event := event105705
    frameStart := 105690 },
  { event := event105706
    frameStart := 105690 },
  { event := event105707
    frameStart := 105690 },
  { event := event105708
    frameStart := 105690 },
  { event := event105709
    frameStart := 105690 },
  { event := event105710
    frameStart := 105690 },
  { event := event105711
    frameStart := 105690 }
]

def eventLeaf6607 : Array AnnotatedEvent := #[
  { event := event105712
    frameStart := 105690 },
  { event := event105713
    frameStart := 105690 },
  { event := event105714
    frameStart := 105690 },
  { event := event105715
    frameStart := 105690 },
  { event := event105716
    frameStart := 105690 },
  { event := event105717
    frameStart := 105690 },
  { event := event105718
    frameStart := 105690 },
  { event := event105719
    frameStart := 105690 },
  { event := event105720
    frameStart := 105690 },
  { event := event105721
    frameStart := 105690 },
  { event := event105722
    frameStart := 105690 },
  { event := event105723
    frameStart := 105690 },
  { event := event105724
    frameStart := 105690 },
  { event := event105725
    frameStart := 105690 },
  { event := event105726
    frameStart := 105690 },
  { event := event105727
    frameStart := 105690 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events412
