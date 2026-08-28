import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events123

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event31488 : Event := .preFoldPolynomial 31487 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact31489RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event31489 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17516⟩⟩) 31488 exact31489RawTerms .large 31486 .exactZero (none)

def event31490 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15719⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨31332, 31490⟩

def event31491 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16421⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (1) 0 2 (.universal 31490 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩) (none) 31489)

def event31492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16421⟩⟩, .relation 31491 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event31493 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16421⟩⟩, .relation 31491 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩, (1)⟩)

def event31494 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16421⟩⟩, .relation 31491 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩, (-1)⟩)

def event31495 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16421⟩⟩, .relation 31491 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact31496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31496RawTermsValid :
    exact31496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16421⟩⟩) exact31496RawTerms .large 31328 (.finite 202072841853861888) (some (31330))

def event31497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17513⟩⟩) 0 ⟨16421⟩ 31496

def event31498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17513⟩⟩) 1 ⟨17512⟩ 31318

def event31499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17513⟩⟩) (.sum [.predecessor 0 31497 .coefficient, .predecessor 1 31498 .coefficient])

def event31500 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17513⟩⟩, .operator (⟨31496, 2⟩, ⟨31318, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16922⟩⟩]⟩, (-1)⟩)

def event31501 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17513⟩⟩, .operator (⟨31496, 0⟩, ⟨31318, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩, (1)⟩)

def event31502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17513⟩⟩) (.sum [.result 31496 .summary, .result 31318 .summary])

def exact31503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31503RawTermsValid :
    exact31503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17513⟩⟩) exact31503RawTerms .large 31499 (.finite 32188807212483706889510625476608) (some (31502))

def event31504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17514⟩⟩) 0 ⟨17513⟩ 31503

def event31505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17514⟩⟩) 1 ⟨7172⟩ 15882

def event31506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17514⟩⟩) (.product (.predecessor 0 31504 .coefficient) (.predecessor 1 31505 .coefficient) (⟨false, false, none, none, none⟩))

def event31507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17514⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event31508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17514⟩⟩) (.product (.result 31503 .summary) (.transfer 31507) (⟨false, false, none, none, none⟩))

def event31509 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17514⟩⟩, .operator (⟨31503, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event31510 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17514⟩⟩, .operator (⟨31503, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event31511 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17514⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event31512 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17514⟩⟩, .relation 31511 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact31513RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31513RawTermsValid :
    exact31513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17514⟩⟩) exact31513RawTerms .large 31506 (.finite 345624685687166110058245054666339432529920) (some (31508))

def event31514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨118⟩⟩) 0 ⟨11⟩ 17049

def event31515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨118⟩⟩) (.identity (.predecessor 0 31514 .coefficient))

def exact31516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩, (1)⟩]

theorem exact31516RawTermsValid :
    exact31516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨118⟩⟩) exact31516RawTerms (.finite 26) 31515 .exactZero (none)

def event31517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7059⟩⟩) 0 ⟨6727⟩ 723

def event31518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7059⟩⟩) 1 ⟨6914⟩ 17057

def event31519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7059⟩⟩) (.tensor (.predecessor 0 31517 .coefficient) (.predecessor 1 31518 .coefficient) true false)

def event31520 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7059⟩⟩, .operator (⟨723, 0⟩, ⟨17057, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact31521RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact31521RawTermsValid :
    exact31521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31521 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7059⟩⟩) exact31521RawTerms .large 31519 .exactZero (none)

def event31522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7610⟩⟩) 0 ⟨5441⟩ 16922

def event31523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7610⟩⟩) 1 ⟨7292⟩ 15896

def event31524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7610⟩⟩) (.product (.predecessor 0 31522 .coefficient) (.predecessor 1 31523 .coefficient) (⟨false, false, none, none, none⟩))

def event31525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7610⟩⟩, .operator (⟨16922, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact31526RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact31526RawTermsValid :
    exact31526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7610⟩⟩) exact31526RawTerms .large 31524 .exactZero (none)

def event31527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9285⟩⟩) 0 ⟨7610⟩ 31526

def event31528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9285⟩⟩) 1 ⟨7059⟩ 31521

def event31529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9285⟩⟩) (.sum [.predecessor 0 31527 .coefficient, .predecessor 1 31528 .coefficient])

def exact31530RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31530RawTermsValid :
    exact31530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9285⟩⟩) exact31530RawTerms .large 31529 .exactZero (none)

def event31531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9286⟩⟩) 0 ⟨9285⟩ 31530

def event31532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9286⟩⟩) 1 ⟨118⟩ 31516

def event31533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9286⟩⟩) (.sum [.predecessor 0 31531 .coefficient, .predecessor 1 31532 .coefficient])

def event31534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9286⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event31535 : Event := .survivorFold (1) 31534

def exact31536RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31536RawTermsValid :
    exact31536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31536 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9286⟩⟩) exact31536RawTerms .large 31533 (.finite 26) (some (31534))

def event31537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9451⟩⟩) 0 ⟨9286⟩ 31536

def event31538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9451⟩⟩) 1 ⟨9286⟩ 31536

def event31539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9451⟩⟩) (.sum [.predecessor 0 31537 .coefficient, .predecessor 1 31538 .coefficient])

def event31540 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9451⟩⟩, .operator (⟨31536, 1⟩, ⟨31536, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event31541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9451⟩⟩, .operator (⟨31536, 0⟩, ⟨31536, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event31542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9451⟩⟩) (.sum [.result 31536 .summary, .result 31536 .summary])

def exact31543RawTerms : List Term := []

theorem exact31543RawTermsValid :
    exact31543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9451⟩⟩) exact31543RawTerms .large 31539 (.finite 52) (some (31542))

def event31544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17515⟩⟩) 0 ⟨9451⟩ 31543

def event31545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17515⟩⟩) 1 ⟨17514⟩ 31513

def event31546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17515⟩⟩) (.sum [.predecessor 0 31544 .coefficient, .predecessor 1 31545 .coefficient])

def event31547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17515⟩⟩) (.sum [.result 31543 .summary, .result 31513 .summary])

def exact31548RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31548RawTermsValid :
    exact31548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17515⟩⟩) exact31548RawTerms .large 31546 (.finite 345624685687166110058245054666339432529972) (some (31547))

def event31549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20380⟩⟩) 0 ⟨17515⟩ 31548

def event31550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20380⟩⟩) 1 ⟨20379⟩ 31301

def event31551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20380⟩⟩) (.sum [.predecessor 0 31549 .coefficient, .predecessor 1 31550 .coefficient])

def event31552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20380⟩⟩) (.sum [.result 31548 .summary, .result 31301 .summary])

def exact31553RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31553RawTermsValid :
    exact31553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31553 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20380⟩⟩) exact31553RawTerms .large 31551 (.finite 691250426059631610003352154589745737891892) (some (31552))

def event31554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23600⟩⟩) 0 ⟨20380⟩ 31553

def event31555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23600⟩⟩) 1 ⟨23599⟩ 31089

def event31556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23600⟩⟩) (.sum [.predecessor 0 31554 .coefficient, .predecessor 1 31555 .coefficient])

def event31557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23600⟩⟩) (.sum [.result 31553 .summary, .result 31089 .summary])

def exact31558RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31558RawTermsValid :
    exact31558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31558 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23600⟩⟩) exact31558RawTerms .large 31556 (.finite 1036877221117396499835321299770218916085812) (some (31557))

def event31559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33620⟩⟩) 0 ⟨23600⟩ 31558

def event31560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33620⟩⟩) 1 ⟨33619⟩ 30877

def event31561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33620⟩⟩) (.sum [.predecessor 0 31559 .coefficient, .predecessor 1 31560 .coefficient])

def event31562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33620⟩⟩) (.sum [.result 31558 .summary, .result 30877 .summary])

def exact31563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31563RawTermsValid :
    exact31563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33620⟩⟩) exact31563RawTerms .large 31561 (.finite 1382506125545760169441014535464825839943732) (some (31562))

def event31564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52680⟩⟩) 0 ⟨33620⟩ 31563

def event31565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52680⟩⟩) 1 ⟨52679⟩ 30665

def event31566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52680⟩⟩) (.sum [.predecessor 0 31564 .coefficient, .predecessor 1 31565 .coefficient])

def event31567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52680⟩⟩) (.sum [.result 31563 .summary, .result 30665 .summary])

def exact31568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31568RawTermsValid :
    exact31568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52680⟩⟩) exact31568RawTerms .large 31566 (.finite 1728139248715321398594155952187700255129652) (some (31567))

def event31569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55660⟩⟩) 0 ⟨52680⟩ 31568

def event31570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55660⟩⟩) 1 ⟨55659⟩ 30453

def event31571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55660⟩⟩) (.sum [.predecessor 0 31569 .coefficient, .predecessor 1 31570 .coefficient])

def event31572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55660⟩⟩) (.sum [.result 31568 .summary, .result 30453 .summary])

def exact31573RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31573RawTermsValid :
    exact31573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31573 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55660⟩⟩) exact31573RawTerms .large 31571 (.finite 2073774481255481407521021459424708415979572) (some (31572))

def event31574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58640⟩⟩) 0 ⟨55660⟩ 31573

def event31575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58640⟩⟩) 1 ⟨58639⟩ 30241

def event31576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58640⟩⟩) (.sum [.predecessor 0 31574 .coefficient, .predecessor 1 31575 .coefficient])

def event31577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58640⟩⟩) (.sum [.result 31573 .summary, .result 30241 .summary])

def exact31578RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31578RawTermsValid :
    exact31578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31578 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58640⟩⟩) exact31578RawTerms .large 31576 (.finite 2419413932536838975995335147689984068157492) (some (31577))

def event31579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61620⟩⟩) 0 ⟨58640⟩ 31578

def event31580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61620⟩⟩) 1 ⟨61619⟩ 30029

def event31581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61620⟩⟩) (.sum [.predecessor 0 31579 .coefficient, .predecessor 1 31580 .coefficient])

def event31582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61620⟩⟩) (.sum [.result 31578 .summary, .result 30029 .summary])

def exact31583RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31583RawTermsValid :
    exact31583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61620⟩⟩) exact31583RawTerms .large 31581 (.finite 2765055493188795324243372926469393465999412) (some (31582))

def event31584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64600⟩⟩) 0 ⟨61620⟩ 31583

def event31585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64600⟩⟩) 1 ⟨64599⟩ 29817

def event31586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64600⟩⟩) (.sum [.predecessor 0 31584 .coefficient, .predecessor 1 31585 .coefficient])

def event31587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64600⟩⟩) (.sum [.result 31583 .summary, .result 29817 .summary])

def exact31588RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31588RawTermsValid :
    exact31588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64600⟩⟩) exact31588RawTerms .large 31586 (.finite 3110701272581949232038858886277070355169332) (some (31587))

def event31589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69481⟩⟩) 0 ⟨64600⟩ 31588

def event31590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69481⟩⟩) 1 ⟨69480⟩ 29605

def event31591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69481⟩⟩) (.sum [.predecessor 0 31589 .coefficient, .predecessor 1 31590 .coefficient])

def event31592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69481⟩⟩) (.sum [.result 31588 .summary, .result 29605 .summary])

def exact31593RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31593RawTermsValid :
    exact31593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69481⟩⟩) exact31593RawTerms .large 31591 (.finite 3456353380086899479155517117627148481331252) (some (31592))

def event31594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69482⟩⟩) 0 ⟨69481⟩ 31593

def event31595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69482⟩⟩) 1 ⟨28069⟩ 29393

def event31596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69482⟩⟩) (.sum [.predecessor 0 31594 .coefficient, .predecessor 1 31595 .coefficient])

def event31597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69482⟩⟩) (.sum [.result 31593 .summary, .result 29393 .summary])

def exact31598RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31598RawTermsValid :
    exact31598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69482⟩⟩) exact31598RawTerms .large 31596 (.finite 3802007596962448506045899439491360353157172) (some (31597))

def event31599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69483⟩⟩) 0 ⟨69482⟩ 31598

def event31600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69483⟩⟩) 1 ⟨30749⟩ 29181

def event31601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69483⟩⟩) (.sum [.predecessor 0 31599 .coefficient, .predecessor 1 31600 .coefficient])

def event31602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69483⟩⟩) (.sum [.result 31598 .summary, .result 29181 .summary])

def exact31603RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31603RawTermsValid :
    exact31603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31603 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69483⟩⟩) exact31603RawTerms .large 31601 (.finite 4147668141949793872257454032897973461975092) (some (31602))

def event31604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69484⟩⟩) 0 ⟨69483⟩ 31603

def event31605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69484⟩⟩) 1 ⟨36409⟩ 28969

def event31606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69484⟩⟩) (.sum [.predecessor 0 31604 .coefficient, .predecessor 1 31605 .coefficient])

def event31607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69484⟩⟩) (.sum [.result 31603 .summary, .result 28969 .summary])

def exact31608RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31608RawTermsValid :
    exact31608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69484⟩⟩) exact31608RawTerms .large 31606 (.finite 4493332905678336798016456807332854062121012) (some (31607))

def event31609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69485⟩⟩) 0 ⟨69484⟩ 31608

def event31610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69485⟩⟩) 1 ⟨39089⟩ 28757

def event31611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69485⟩⟩) (.sum [.predecessor 0 31609 .coefficient, .predecessor 1 31610 .coefficient])

def event31612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69485⟩⟩) (.sum [.result 31608 .summary, .result 28757 .summary])

def exact31613RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31613RawTermsValid :
    exact31613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31613 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69485⟩⟩) exact31613RawTerms .large 31611 (.finite 4838999778777478503549183672281868407930932) (some (31612))

def event31614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69486⟩⟩) 0 ⟨69485⟩ 31613

def event31615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69486⟩⟩) 1 ⟨41769⟩ 28545

def event31616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69486⟩⟩) (.sum [.predecessor 0 31614 .coefficient, .predecessor 1 31615 .coefficient])

def event31617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69486⟩⟩) (.sum [.result 31613 .summary, .result 28545 .summary])

def exact31618RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31618RawTermsValid :
    exact31618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69486⟩⟩) exact31618RawTerms .large 31616 (.finite 5184670870617817768629358718259150245068852) (some (31617))

def event31619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69487⟩⟩) 0 ⟨69486⟩ 31618

def event31620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69487⟩⟩) 1 ⟨44449⟩ 28333

def event31621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69487⟩⟩) (.sum [.predecessor 0 31619 .coefficient, .predecessor 1 31620 .coefficient])

def event31622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69487⟩⟩) (.sum [.result 31618 .summary, .result 28333 .summary])

def exact31623RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31623RawTermsValid :
    exact31623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69487⟩⟩) exact31623RawTerms .large 31621 (.finite 5530348290569953373030706035778833319198772) (some (31622))

def event31624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69488⟩⟩) 0 ⟨69487⟩ 31623

def event31625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69488⟩⟩) 1 ⟨47129⟩ 28121

def event31626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69488⟩⟩) (.sum [.predecessor 0 31624 .coefficient, .predecessor 1 31625 .coefficient])

def event31627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69488⟩⟩) (.sum [.result 31623 .summary, .result 28121 .summary])

def exact31628RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31628RawTermsValid :
    exact31628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69488⟩⟩) exact31628RawTerms .large 31626 (.finite 5876032038633885316753225624840917630320692) (some (31627))

def event31629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69489⟩⟩) 0 ⟨69488⟩ 31628

def event31630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69489⟩⟩) 1 ⟨49809⟩ 27909

def event31631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69489⟩⟩) (.sum [.predecessor 0 31629 .coefficient, .predecessor 1 31630 .coefficient])

def event31632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69489⟩⟩) (.sum [.result 31628 .summary, .result 27909 .summary])

def exact31633RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31633RawTermsValid :
    exact31633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69489⟩⟩) exact31633RawTerms .large 31631 (.finite 6221717896068416040249469304417135687106612) (some (31632))

def event31634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70974⟩⟩) 0 ⟨69489⟩ 31633

def event31635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70974⟩⟩) 1 ⟨70972⟩ 27697

def event31636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70974⟩⟩) (.sum [.predecessor 0 31634 .coefficient, .predecessor 1 31635 .coefficient])

def event31637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70974⟩⟩) (.sum [.result 31633 .summary, .result 27697 .summary])

def exact31638RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact31638RawTermsValid :
    exact31638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70974⟩⟩) exact31638RawTerms .large 31636 (.finite 66805187227601152574551644069558752530002096506798132) (some (31637))

def event31639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32⟩⟩) (.authority (.operator))

def exact31640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32⟩⟩]⟩, (1)⟩]

theorem exact31640RawTermsValid :
    exact31640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32⟩⟩) exact31640RawTerms (.finite 26) 31639 .exactZero (none)

def event31641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7401⟩⟩) 0 ⟨2377⟩ 27

def event31642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7401⟩⟩) 1 ⟨7234⟩ 15987

def event31643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7401⟩⟩) (.product (.predecessor 0 31641 .coefficient) (.predecessor 1 31642 .coefficient) (⟨false, false, none, none, none⟩))

def event31644 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7401⟩⟩, .operator (⟨27, 0⟩, ⟨15987, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩]⟩, (1)⟩)

def exact31645RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩]⟩, (1)⟩]

theorem exact31645RawTermsValid :
    exact31645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7401⟩⟩) exact31645RawTerms .large 31643 .exactZero (none)

def event31646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9215⟩⟩) 0 ⟨7401⟩ 31645

def event31647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9215⟩⟩) 1 ⟨6914⟩ 17057

def event31648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9215⟩⟩) (.sum [.predecessor 0 31646 .coefficient, .predecessor 1 31647 .coefficient])

def exact31649RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩]⟩, (1)⟩]

theorem exact31649RawTermsValid :
    exact31649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31649 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9215⟩⟩) exact31649RawTerms .large 31648 .exactZero (none)

def event31650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9216⟩⟩) 0 ⟨9215⟩ 31649

def event31651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9216⟩⟩) 1 ⟨32⟩ 31640

def event31652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9216⟩⟩) (.sum [.predecessor 0 31650 .coefficient, .predecessor 1 31651 .coefficient])

def event31653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9216⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32⟩⟩]⟩) [⟨.result 31640 .coefficient, false, none⟩])

def event31654 : Event := .survivorFold (1) 31653

def exact31655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩]⟩, (1)⟩]

theorem exact31655RawTermsValid :
    exact31655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9216⟩⟩) exact31655RawTerms .large 31652 (.finite 26) (some (31653))

def event31656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9618⟩⟩) 0 ⟨9216⟩ 31655

def event31657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9618⟩⟩) 1 ⟨9584⟩ 15984

def event31658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9618⟩⟩) (.product (.predecessor 0 31656 .coefficient) (.predecessor 1 31657 .coefficient) (⟨false, false, none, none, none⟩))

def event31659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9618⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event31660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9618⟩⟩) (.product (.result 31655 .summary) (.transfer 31659) (⟨false, false, none, none, none⟩))

def event31661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .operator (⟨31655, 0⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event31662 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨9618⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event31663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 18, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event31664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 17, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event31665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 16, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event31666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 15, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event31667 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 14, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event31668 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 13, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event31669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 12, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event31670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 11, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event31671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 10, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event31672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 9, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩)

def event31673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 8, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event31674 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 7, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩)

def event31675 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 6, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event31676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 5, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event31677 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 4, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def event31678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩)

def event31679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event31680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩)

def event31681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .relation 31662 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event31682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9618⟩⟩, .operator (⟨31655, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩)

def exact31683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩]

theorem exact31683RawTermsValid :
    exact31683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9618⟩⟩) exact31683RawTerms .large 31658 (.finite 279172874240) (some (31660))

def event31684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70975⟩⟩) 0 ⟨9618⟩ 31683

def event31685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70975⟩⟩) 1 ⟨70974⟩ 31638

def event31686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70975⟩⟩) (.sum [.predecessor 0 31684 .coefficient, .predecessor 1 31685 .coefficient])

def event31687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 18⟩, ⟨31638, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event31688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 17⟩, ⟨31638, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event31689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 16⟩, ⟨31638, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event31690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 15⟩, ⟨31638, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event31691 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 14⟩, ⟨31638, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event31692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 13⟩, ⟨31638, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event31693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 12⟩, ⟨31638, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event31694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 11⟩, ⟨31638, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def event31695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 10⟩, ⟨31638, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩)

def event31696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 9⟩, ⟨31638, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩)

def event31697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 8⟩, ⟨31638, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩)

def event31698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 7⟩, ⟨31638, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩)

def event31699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 6⟩, ⟨31638, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩)

def event31700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 5⟩, ⟨31638, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩)

def event31701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 4⟩, ⟨31638, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩)

def event31702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 3⟩, ⟨31638, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩)

def event31703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 2⟩, ⟨31638, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event31704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 1⟩, ⟨31638, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩)

def event31705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70975⟩⟩, .operator (⟨31683, 0⟩, ⟨31638, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event31706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70975⟩⟩) (.sum [.result 31683 .summary, .result 31638 .summary])

def exact31707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩]

theorem exact31707RawTermsValid :
    exact31707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event31707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70975⟩⟩) exact31707RawTerms .large 31686 (.finite 66805187227601152574551644069558752530002375679672372) (some (31706))

def event31708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70976⟩⟩) 0 ⟨70975⟩ 31707

def event31709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70976⟩⟩) 1 ⟨9492⟩ 15510

def event31710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70976⟩⟩) (.product (.predecessor 0 31708 .coefficient) (.predecessor 1 31709 .coefficient) (⟨false, false, none, none, none⟩))

def event31711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70976⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) [⟨.result 15506 .coefficient, false, none⟩])

def event31712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70976⟩⟩) (.product (.result 31707 .summary) (.transfer 31711) (⟨false, false, none, none, none⟩))

def event31713 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 5⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (1)⟩)

def event31714 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31715 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31714 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event31716 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 7⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31717 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31717 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 8⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31720 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31720 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 9⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31723 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31723 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 11⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31726 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31727 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31726 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31728 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 12⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31729 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31729 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 13⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31732 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31732 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31734 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 15⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31735 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31736 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31735 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31737 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 16⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31738 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31738 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 18⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def event31741 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70976⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9491⟩⟩) ⟨7235⟩ 15503)

def event31742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .relation 31741 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event31743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70976⟩⟩, .operator (⟨31707, 0⟩, ⟨15510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩]⟩, (-1)⟩)

def eventLeaf1968 : Array AnnotatedEvent := #[
  { event := event31488
    frameStart := 31386 },
  { event := event31489
    frameStart := 31386 },
  { event := event31490
    frameStart := 0 },
  { event := event31491
    frameStart := 0 },
  { event := event31492
    frameStart := 0 },
  { event := event31493
    frameStart := 0 },
  { event := event31494
    frameStart := 0 },
  { event := event31495
    frameStart := 0 },
  { event := event31496
    frameStart := 0 },
  { event := event31497
    frameStart := 0 },
  { event := event31498
    frameStart := 0 },
  { event := event31499
    frameStart := 0 },
  { event := event31500
    frameStart := 0 },
  { event := event31501
    frameStart := 0 },
  { event := event31502
    frameStart := 0 },
  { event := event31503
    frameStart := 0 }
]

def eventLeaf1969 : Array AnnotatedEvent := #[
  { event := event31504
    frameStart := 0 },
  { event := event31505
    frameStart := 0 },
  { event := event31506
    frameStart := 0 },
  { event := event31507
    frameStart := 0 },
  { event := event31508
    frameStart := 0 },
  { event := event31509
    frameStart := 0 },
  { event := event31510
    frameStart := 0 },
  { event := event31511
    frameStart := 0 },
  { event := event31512
    frameStart := 0 },
  { event := event31513
    frameStart := 0 },
  { event := event31514
    frameStart := 0 },
  { event := event31515
    frameStart := 0 },
  { event := event31516
    frameStart := 0 },
  { event := event31517
    frameStart := 0 },
  { event := event31518
    frameStart := 0 },
  { event := event31519
    frameStart := 0 }
]

def eventLeaf1970 : Array AnnotatedEvent := #[
  { event := event31520
    frameStart := 0 },
  { event := event31521
    frameStart := 0 },
  { event := event31522
    frameStart := 0 },
  { event := event31523
    frameStart := 0 },
  { event := event31524
    frameStart := 0 },
  { event := event31525
    frameStart := 0 },
  { event := event31526
    frameStart := 0 },
  { event := event31527
    frameStart := 0 },
  { event := event31528
    frameStart := 0 },
  { event := event31529
    frameStart := 0 },
  { event := event31530
    frameStart := 0 },
  { event := event31531
    frameStart := 0 },
  { event := event31532
    frameStart := 0 },
  { event := event31533
    frameStart := 0 },
  { event := event31534
    frameStart := 0 },
  { event := event31535
    frameStart := 0 }
]

def eventLeaf1971 : Array AnnotatedEvent := #[
  { event := event31536
    frameStart := 0 },
  { event := event31537
    frameStart := 0 },
  { event := event31538
    frameStart := 0 },
  { event := event31539
    frameStart := 0 },
  { event := event31540
    frameStart := 0 },
  { event := event31541
    frameStart := 0 },
  { event := event31542
    frameStart := 0 },
  { event := event31543
    frameStart := 0 },
  { event := event31544
    frameStart := 0 },
  { event := event31545
    frameStart := 0 },
  { event := event31546
    frameStart := 0 },
  { event := event31547
    frameStart := 0 },
  { event := event31548
    frameStart := 0 },
  { event := event31549
    frameStart := 0 },
  { event := event31550
    frameStart := 0 },
  { event := event31551
    frameStart := 0 }
]

def eventLeaf1972 : Array AnnotatedEvent := #[
  { event := event31552
    frameStart := 0 },
  { event := event31553
    frameStart := 0 },
  { event := event31554
    frameStart := 0 },
  { event := event31555
    frameStart := 0 },
  { event := event31556
    frameStart := 0 },
  { event := event31557
    frameStart := 0 },
  { event := event31558
    frameStart := 0 },
  { event := event31559
    frameStart := 0 },
  { event := event31560
    frameStart := 0 },
  { event := event31561
    frameStart := 0 },
  { event := event31562
    frameStart := 0 },
  { event := event31563
    frameStart := 0 },
  { event := event31564
    frameStart := 0 },
  { event := event31565
    frameStart := 0 },
  { event := event31566
    frameStart := 0 },
  { event := event31567
    frameStart := 0 }
]

def eventLeaf1973 : Array AnnotatedEvent := #[
  { event := event31568
    frameStart := 0 },
  { event := event31569
    frameStart := 0 },
  { event := event31570
    frameStart := 0 },
  { event := event31571
    frameStart := 0 },
  { event := event31572
    frameStart := 0 },
  { event := event31573
    frameStart := 0 },
  { event := event31574
    frameStart := 0 },
  { event := event31575
    frameStart := 0 },
  { event := event31576
    frameStart := 0 },
  { event := event31577
    frameStart := 0 },
  { event := event31578
    frameStart := 0 },
  { event := event31579
    frameStart := 0 },
  { event := event31580
    frameStart := 0 },
  { event := event31581
    frameStart := 0 },
  { event := event31582
    frameStart := 0 },
  { event := event31583
    frameStart := 0 }
]

def eventLeaf1974 : Array AnnotatedEvent := #[
  { event := event31584
    frameStart := 0 },
  { event := event31585
    frameStart := 0 },
  { event := event31586
    frameStart := 0 },
  { event := event31587
    frameStart := 0 },
  { event := event31588
    frameStart := 0 },
  { event := event31589
    frameStart := 0 },
  { event := event31590
    frameStart := 0 },
  { event := event31591
    frameStart := 0 },
  { event := event31592
    frameStart := 0 },
  { event := event31593
    frameStart := 0 },
  { event := event31594
    frameStart := 0 },
  { event := event31595
    frameStart := 0 },
  { event := event31596
    frameStart := 0 },
  { event := event31597
    frameStart := 0 },
  { event := event31598
    frameStart := 0 },
  { event := event31599
    frameStart := 0 }
]

def eventLeaf1975 : Array AnnotatedEvent := #[
  { event := event31600
    frameStart := 0 },
  { event := event31601
    frameStart := 0 },
  { event := event31602
    frameStart := 0 },
  { event := event31603
    frameStart := 0 },
  { event := event31604
    frameStart := 0 },
  { event := event31605
    frameStart := 0 },
  { event := event31606
    frameStart := 0 },
  { event := event31607
    frameStart := 0 },
  { event := event31608
    frameStart := 0 },
  { event := event31609
    frameStart := 0 },
  { event := event31610
    frameStart := 0 },
  { event := event31611
    frameStart := 0 },
  { event := event31612
    frameStart := 0 },
  { event := event31613
    frameStart := 0 },
  { event := event31614
    frameStart := 0 },
  { event := event31615
    frameStart := 0 }
]

def eventLeaf1976 : Array AnnotatedEvent := #[
  { event := event31616
    frameStart := 0 },
  { event := event31617
    frameStart := 0 },
  { event := event31618
    frameStart := 0 },
  { event := event31619
    frameStart := 0 },
  { event := event31620
    frameStart := 0 },
  { event := event31621
    frameStart := 0 },
  { event := event31622
    frameStart := 0 },
  { event := event31623
    frameStart := 0 },
  { event := event31624
    frameStart := 0 },
  { event := event31625
    frameStart := 0 },
  { event := event31626
    frameStart := 0 },
  { event := event31627
    frameStart := 0 },
  { event := event31628
    frameStart := 0 },
  { event := event31629
    frameStart := 0 },
  { event := event31630
    frameStart := 0 },
  { event := event31631
    frameStart := 0 }
]

def eventLeaf1977 : Array AnnotatedEvent := #[
  { event := event31632
    frameStart := 0 },
  { event := event31633
    frameStart := 0 },
  { event := event31634
    frameStart := 0 },
  { event := event31635
    frameStart := 0 },
  { event := event31636
    frameStart := 0 },
  { event := event31637
    frameStart := 0 },
  { event := event31638
    frameStart := 0 },
  { event := event31639
    frameStart := 0 },
  { event := event31640
    frameStart := 0 },
  { event := event31641
    frameStart := 0 },
  { event := event31642
    frameStart := 0 },
  { event := event31643
    frameStart := 0 },
  { event := event31644
    frameStart := 0 },
  { event := event31645
    frameStart := 0 },
  { event := event31646
    frameStart := 0 },
  { event := event31647
    frameStart := 0 }
]

def eventLeaf1978 : Array AnnotatedEvent := #[
  { event := event31648
    frameStart := 0 },
  { event := event31649
    frameStart := 0 },
  { event := event31650
    frameStart := 0 },
  { event := event31651
    frameStart := 0 },
  { event := event31652
    frameStart := 0 },
  { event := event31653
    frameStart := 0 },
  { event := event31654
    frameStart := 0 },
  { event := event31655
    frameStart := 0 },
  { event := event31656
    frameStart := 0 },
  { event := event31657
    frameStart := 0 },
  { event := event31658
    frameStart := 0 },
  { event := event31659
    frameStart := 0 },
  { event := event31660
    frameStart := 0 },
  { event := event31661
    frameStart := 0 },
  { event := event31662
    frameStart := 0 },
  { event := event31663
    frameStart := 0 }
]

def eventLeaf1979 : Array AnnotatedEvent := #[
  { event := event31664
    frameStart := 0 },
  { event := event31665
    frameStart := 0 },
  { event := event31666
    frameStart := 0 },
  { event := event31667
    frameStart := 0 },
  { event := event31668
    frameStart := 0 },
  { event := event31669
    frameStart := 0 },
  { event := event31670
    frameStart := 0 },
  { event := event31671
    frameStart := 0 },
  { event := event31672
    frameStart := 0 },
  { event := event31673
    frameStart := 0 },
  { event := event31674
    frameStart := 0 },
  { event := event31675
    frameStart := 0 },
  { event := event31676
    frameStart := 0 },
  { event := event31677
    frameStart := 0 },
  { event := event31678
    frameStart := 0 },
  { event := event31679
    frameStart := 0 }
]

def eventLeaf1980 : Array AnnotatedEvent := #[
  { event := event31680
    frameStart := 0 },
  { event := event31681
    frameStart := 0 },
  { event := event31682
    frameStart := 0 },
  { event := event31683
    frameStart := 0 },
  { event := event31684
    frameStart := 0 },
  { event := event31685
    frameStart := 0 },
  { event := event31686
    frameStart := 0 },
  { event := event31687
    frameStart := 0 },
  { event := event31688
    frameStart := 0 },
  { event := event31689
    frameStart := 0 },
  { event := event31690
    frameStart := 0 },
  { event := event31691
    frameStart := 0 },
  { event := event31692
    frameStart := 0 },
  { event := event31693
    frameStart := 0 },
  { event := event31694
    frameStart := 0 },
  { event := event31695
    frameStart := 0 }
]

def eventLeaf1981 : Array AnnotatedEvent := #[
  { event := event31696
    frameStart := 0 },
  { event := event31697
    frameStart := 0 },
  { event := event31698
    frameStart := 0 },
  { event := event31699
    frameStart := 0 },
  { event := event31700
    frameStart := 0 },
  { event := event31701
    frameStart := 0 },
  { event := event31702
    frameStart := 0 },
  { event := event31703
    frameStart := 0 },
  { event := event31704
    frameStart := 0 },
  { event := event31705
    frameStart := 0 },
  { event := event31706
    frameStart := 0 },
  { event := event31707
    frameStart := 0 },
  { event := event31708
    frameStart := 0 },
  { event := event31709
    frameStart := 0 },
  { event := event31710
    frameStart := 0 },
  { event := event31711
    frameStart := 0 }
]

def eventLeaf1982 : Array AnnotatedEvent := #[
  { event := event31712
    frameStart := 0 },
  { event := event31713
    frameStart := 0 },
  { event := event31714
    frameStart := 0 },
  { event := event31715
    frameStart := 0 },
  { event := event31716
    frameStart := 0 },
  { event := event31717
    frameStart := 0 },
  { event := event31718
    frameStart := 0 },
  { event := event31719
    frameStart := 0 },
  { event := event31720
    frameStart := 0 },
  { event := event31721
    frameStart := 0 },
  { event := event31722
    frameStart := 0 },
  { event := event31723
    frameStart := 0 },
  { event := event31724
    frameStart := 0 },
  { event := event31725
    frameStart := 0 },
  { event := event31726
    frameStart := 0 },
  { event := event31727
    frameStart := 0 }
]

def eventLeaf1983 : Array AnnotatedEvent := #[
  { event := event31728
    frameStart := 0 },
  { event := event31729
    frameStart := 0 },
  { event := event31730
    frameStart := 0 },
  { event := event31731
    frameStart := 0 },
  { event := event31732
    frameStart := 0 },
  { event := event31733
    frameStart := 0 },
  { event := event31734
    frameStart := 0 },
  { event := event31735
    frameStart := 0 },
  { event := event31736
    frameStart := 0 },
  { event := event31737
    frameStart := 0 },
  { event := event31738
    frameStart := 0 },
  { event := event31739
    frameStart := 0 },
  { event := event31740
    frameStart := 0 },
  { event := event31741
    frameStart := 0 },
  { event := event31742
    frameStart := 0 },
  { event := event31743
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events123
