import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events111

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event28416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15436⟩⟩) 1 ⟨15434⟩ 28414

def event28417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15436⟩⟩) (.product (.predecessor 0 28415 .coefficient) (.predecessor 1 28416 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28418 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15436⟩⟩, .operator (⟨28371, 0⟩, ⟨28414, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28419RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28419RawTermsValid :
    exact28419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15436⟩⟩) exact28419RawTerms .large 28417 .exactZero (none)

def event28420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 28353

def event28421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact28422RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact28422RawTermsValid :
    exact28422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact28422RawTerms .large 28421 .exactZero (none)

def event28423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15437⟩⟩) 0 ⟨6693⟩ 28422

def event28424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15437⟩⟩) 1 ⟨15436⟩ 28419

def event28425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15437⟩⟩) (.sum [.predecessor 0 28423 .coefficient, .predecessor 1 28424 .coefficient])

def exact28426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28426RawTermsValid :
    exact28426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15437⟩⟩) exact28426RawTerms .large 28425 .exactZero (none)

def event28427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25315⟩⟩) 0 ⟨15437⟩ 28426

def event28428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25315⟩⟩) 1 ⟨25314⟩ 28411

def event28429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25315⟩⟩) (.sum [.predecessor 0 28427 .coefficient, .predecessor 1 28428 .coefficient])

def exact28430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28430RawTermsValid :
    exact28430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25315⟩⟩) exact28430RawTerms .large 28429 .exactZero (none)

def event28431 : Event := .preFoldPolynomial 28430 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event28432 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25315⟩⟩) 28431 exact28432RawTerms .large 28429 .exactZero (none)

def event28433 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12192⟩⟩) ⟨⟨106⟩, ⟨10⟩, ⟨109⟩⟩ ⟨28267, 28433⟩

def event28434 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19255⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (1) 0 2 (.universal 28433 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) (none) 28432)

def event28435 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19255⟩⟩, .relation 28434 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩)

def event28436 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19255⟩⟩, .relation 28434 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩)

def event28437 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19255⟩⟩, .relation 28434 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩)

def event28438 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19255⟩⟩, .relation 28434 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact28439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28439RawTermsValid :
    exact28439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19255⟩⟩) exact28439RawTerms .large 28263 (.finite 1811303510016) (some (28265))

def event28440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25313⟩⟩) 0 ⟨19255⟩ 28439

def event28441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25313⟩⟩) 1 ⟨25312⟩ 28253

def event28442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25313⟩⟩) (.sum [.predecessor 0 28440 .coefficient, .predecessor 1 28441 .coefficient])

def event28443 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25313⟩⟩, .operator (⟨28439, 2⟩, ⟨28253, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (-1)⟩)

def event28444 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25313⟩⟩, .operator (⟨28439, 1⟩, ⟨28253, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩)

def event28445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25313⟩⟩) (.sum [.result 28439 .summary, .result 28253 .summary])

def exact28446RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28446RawTermsValid :
    exact28446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25313⟩⟩) exact28446RawTerms .large 28442 (.finite 352024077676544) (some (28445))

def event28447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27039⟩⟩) 0 ⟨25313⟩ 28446

def event28448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27039⟩⟩) 1 ⟨27037⟩ 28169

def event28449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27039⟩⟩) (.product (.predecessor 0 28447 .coefficient) (.predecessor 1 28448 .coefficient) (⟨false, false, none, none, none⟩))

def event28450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27039⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩) [⟨.result 28169 .coefficient, false, none⟩])

def event28451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27039⟩⟩) (.product (.result 28446 .summary) (.transfer 28450) (⟨false, false, none, none, none⟩))

def event28452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27039⟩⟩, .operator (⟨28446, 0⟩, ⟨28169, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩)

def event28453 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27039⟩⟩, .operator (⟨28446, 1⟩, ⟨28169, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩)

def event28454 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27039⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27037⟩⟩) ⟨23919⟩ 28166)

def event28455 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27039⟩⟩, .relation 28454 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (-1)⟩)

def exact28456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (-1)⟩]

theorem exact28456RawTermsValid :
    exact28456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27039⟩⟩) exact28456RawTerms .large 28449 (.finite 1291933997458159304704) (some (28451))

def event28457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20836⟩⟩) 0 ⟨15435⟩ 1181

def event28458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20836⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact28459RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩]

theorem exact28459RawTermsValid :
    exact28459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20836⟩⟩) exact28459RawTerms (.finite 136065468) 28458 .exactZero (none)

def event28460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20838⟩⟩) 0 ⟨20836⟩ 28459

def event28461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20838⟩⟩) 1 ⟨2348⟩ 4

def event28462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20838⟩⟩) (.scale (.predecessor 0 28460 .coefficient) (.value (.predecessor 1 28461 .coefficient)))

def exact28463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩]

theorem exact28463RawTermsValid :
    exact28463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28463 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20838⟩⟩) exact28463RawTerms (.finite 136065468) 28462 .exactZero (none)

def event28464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20839⟩⟩) 0 ⟨5559⟩ 21512

def event28465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20839⟩⟩) 1 ⟨20838⟩ 28463

def event28466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20839⟩⟩) (.product (.predecessor 0 28464 .coefficient) (.predecessor 1 28465 .coefficient) (⟨false, false, none, none, none⟩))

def event28467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20839⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩) [⟨.result 28459 .coefficient, false, none⟩])

def event28468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20839⟩⟩) (.product (.result 21512 .summary) (.transfer 28467) (⟨false, false, none, none, none⟩))

def event28469 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20839⟩⟩, .operator (⟨21512, 0⟩, ⟨28463, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩)

def event28470 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20837⟩⟩)

def event28471 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event28472 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event28473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event28474 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event28475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event28476 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event28477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event28478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event28479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 28478

def event28480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 28476

def event28481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 28479 .coefficient) (.value (.predecessor 1 28480 .coefficient)))

def event28482 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28482

def event28484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 28474

def event28485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28483 .coefficient, .predecessor 1 28484 .coefficient])

def event28486 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28487 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28486

def event28488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 28472

def event28489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28488 .coefficient))

def event28490 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11145⟩⟩) 0 ⟨5554⟩ 28490

def event28492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11145⟩⟩) (.authority (.programFamilyFact))

def exact28493RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩], []⟩, (1)⟩]

theorem exact28493RawTermsValid :
    exact28493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28493 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11145⟩⟩) exact28493RawTerms (.finite 6) 28492 .exactZero (none)

def event28494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12190⟩⟩) 0 ⟨5554⟩ 28490

def event28495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12190⟩⟩) (.authority (.programFamilyFact))

def exact28496RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28496RawTermsValid :
    exact28496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12190⟩⟩) exact28496RawTerms (.finite 6) 28495 .exactZero (none)

def event28497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 0 ⟨12190⟩ 28496

def event28498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 1 ⟨11145⟩ 28493

def event28499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.product (.predecessor 0 28497 .coefficient) (.predecessor 1 28498 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩) [⟨.result 28496 .coefficient, true, some 1⟩, ⟨.result 28493 .coefficient, true, some 1⟩])

def event28501 : Event := .survivorFold (1) 28500

def exact28502RawTerms : List Term := []

theorem exact28502RawTermsValid :
    exact28502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12191⟩⟩) exact28502RawTerms (.finite 36) 28499 (.finite 36) (some (28500))

def event28503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12192⟩⟩) 0 ⟨12191⟩ 28502

def event28504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.identity (.predecessor 0 28503 .coefficient))

def event28505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.finite 36)

def event28506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15434⟩⟩) 0 ⟨12192⟩ 28505

def event28507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15434⟩⟩) (.authority (.programFamilyFact))

def exact28508RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], []⟩, (1)⟩]

theorem exact28508RawTermsValid :
    exact28508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28508 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15434⟩⟩) exact28508RawTerms (.finite 6) 28507 .exactZero (none)

def event28509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15435⟩⟩) 0 ⟨15434⟩ 28508

def event28510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15435⟩⟩) (.identity (.predecessor 0 28509 .coefficient))

def event28511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15435⟩⟩) (.finite 6)

def event28512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20836⟩⟩) 0 ⟨15435⟩ 28511

def event28513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20836⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact28514RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩]

theorem exact28514RawTermsValid :
    exact28514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28514 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20836⟩⟩) exact28514RawTerms (.finite 136065468) 28513 .exactZero (none)

def event28515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact28516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact28516RawTermsValid :
    exact28516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28516 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact28516RawTerms .large 28515 .exactZero (none)

def event28517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20837⟩⟩) 0 ⟨6⟩ 28516

def event28518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20837⟩⟩) 1 ⟨20836⟩ 28514

def event28519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20837⟩⟩) (.product (.predecessor 0 28517 .coefficient) (.predecessor 1 28518 .coefficient) (⟨false, false, none, none, none⟩))

def event28520 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20837⟩⟩, .operator (⟨28516, 0⟩, ⟨28514, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩)

def exact28521RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩]

theorem exact28521RawTermsValid :
    exact28521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20837⟩⟩) exact28521RawTerms .large 28519 .exactZero (none)

def event28522 : Event := .preFoldPolynomial 28521 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩] .exactZero none

def exact28523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩, (1)⟩]

def event28523 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20837⟩⟩) 28522 exact28523RawTerms .large 28519 .exactZero (none)

def event28524 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27042⟩⟩)

def event28525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event28526 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event28527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event28528 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event28529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event28530 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event28531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event28532 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event28533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 28532

def event28534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 28530

def event28535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 28533 .coefficient) (.value (.predecessor 1 28534 .coefficient)))

def event28536 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28536

def event28538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 28528

def event28539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28537 .coefficient, .predecessor 1 28538 .coefficient])

def event28540 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28540

def event28542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 28526

def event28543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28542 .coefficient))

def event28544 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11145⟩⟩) 0 ⟨5554⟩ 28544

def event28546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11145⟩⟩) (.authority (.programFamilyFact))

def exact28547RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩], []⟩, (1)⟩]

theorem exact28547RawTermsValid :
    exact28547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28547 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11145⟩⟩) exact28547RawTerms (.finite 6) 28546 .exactZero (none)

def event28548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12190⟩⟩) 0 ⟨5554⟩ 28544

def event28549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12190⟩⟩) (.authority (.programFamilyFact))

def exact28550RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28550RawTermsValid :
    exact28550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28550 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12190⟩⟩) exact28550RawTerms (.finite 6) 28549 .exactZero (none)

def event28551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 0 ⟨12190⟩ 28550

def event28552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 1 ⟨11145⟩ 28547

def event28553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.product (.predecessor 0 28551 .coefficient) (.predecessor 1 28552 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28554 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12191⟩⟩, .operator (⟨28550, 0⟩, ⟨28547, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩)

def exact28555RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28555RawTermsValid :
    exact28555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28555 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12191⟩⟩) exact28555RawTerms (.finite 36) 28553 .exactZero (none)

def event28556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12192⟩⟩) 0 ⟨12191⟩ 28555

def event28557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.identity (.predecessor 0 28556 .coefficient))

def event28558 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.finite 36)

def event28559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15434⟩⟩) 0 ⟨12192⟩ 28558

def event28560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15434⟩⟩) (.authority (.programFamilyFact))

def exact28561RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], []⟩, (1)⟩]

theorem exact28561RawTermsValid :
    exact28561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15434⟩⟩) exact28561RawTerms (.finite 6) 28560 .exactZero (none)

def event28562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15435⟩⟩) 0 ⟨15434⟩ 28561

def event28563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15435⟩⟩) (.identity (.predecessor 0 28562 .coefficient))

def event28564 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15435⟩⟩) (.finite 6)

def event28565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23917⟩⟩) 0 ⟨15435⟩ 28564

def event28566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23917⟩⟩) (.authority (.programFamilyFact))

def event28567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23917⟩⟩) (.finite 3720)

def event28568 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event28569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23919⟩⟩) 0 ⟨6689⟩ 28568

def event28570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23919⟩⟩) 1 ⟨23917⟩ 28567

def event28571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23919⟩⟩) (.authority (.operator))

def exact28572RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩]

theorem exact28572RawTermsValid :
    exact28572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28572 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23919⟩⟩) exact28572RawTerms .large 28571 .exactZero (none)

def event28573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27037⟩⟩) 0 ⟨23919⟩ 28572

def event28574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27037⟩⟩) (.authority (.operator))

def exact28575RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩]

theorem exact28575RawTermsValid :
    exact28575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28575 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27037⟩⟩) exact28575RawTerms (.finite 8192) 28574 .exactZero (none)

def event28576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event28577 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event28578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15474⟩⟩) 0 ⟨15435⟩ 28564

def event28579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15474⟩⟩) 1 ⟨110⟩ 28577

def event28580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15474⟩⟩) (.sum [.predecessor 0 28578 .coefficient, .predecessor 1 28579 .coefficient])

def event28581 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15474⟩⟩) (.finite 6)

def event28582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15475⟩⟩) 0 ⟨15474⟩ 28581

def event28583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15475⟩⟩) (.identity (.predecessor 0 28582 .coefficient))

def exact28584RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], []⟩, (1)⟩]

theorem exact28584RawTermsValid :
    exact28584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15475⟩⟩) exact28584RawTerms (.finite 6) 28583 .exactZero (none)

def event28585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact28586RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28586RawTermsValid :
    exact28586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28586 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact28586RawTerms .large 28585 .exactZero (none)

def event28587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15476⟩⟩) 0 ⟨6544⟩ 28586

def event28588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15476⟩⟩) 1 ⟨15475⟩ 28584

def event28589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15476⟩⟩) (.product (.predecessor 0 28587 .coefficient) (.predecessor 1 28588 .coefficient) (⟨false, false, none, none, none⟩))

def event28590 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15476⟩⟩, .operator (⟨28586, 0⟩, ⟨28584, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28591RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28591RawTermsValid :
    exact28591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28591 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15476⟩⟩) exact28591RawTerms .large 28589 .exactZero (none)

def event28592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 28568

def event28593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact28594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact28594RawTermsValid :
    exact28594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28594 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact28594RawTerms .large 28593 .exactZero (none)

def event28595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15477⟩⟩) 0 ⟨6693⟩ 28594

def event28596 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15477⟩⟩) 1 ⟨15476⟩ 28591

def event28597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15477⟩⟩) (.sum [.predecessor 0 28595 .coefficient, .predecessor 1 28596 .coefficient])

def exact28598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28598RawTermsValid :
    exact28598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28598 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15477⟩⟩) exact28598RawTerms .large 28597 .exactZero (none)

def event28599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27038⟩⟩) 0 ⟨15477⟩ 28598

def event28600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27038⟩⟩) 1 ⟨27037⟩ 28575

def event28601 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27038⟩⟩) (.product (.predecessor 0 28599 .coefficient) (.predecessor 1 28600 .coefficient) (⟨false, false, none, none, none⟩))

def event28602 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27038⟩⟩, .operator (⟨28598, 0⟩, ⟨28575, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩)

def event28603 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27038⟩⟩, .operator (⟨28598, 1⟩, ⟨28575, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩)

def event28604 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27038⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27037⟩⟩) ⟨23919⟩ 28572)

def event28605 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27038⟩⟩, .relation 28604 0, ⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (-1)⟩)

def exact28606RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (-1)⟩]

theorem exact28606RawTermsValid :
    exact28606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28606 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27038⟩⟩) exact28606RawTerms .large 28601 .exactZero (none)

def event28607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17354⟩⟩) 0 ⟨15435⟩ 28564

def event28608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17354⟩⟩) (.authority (.programFamilyFact))

def exact28609RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], []⟩, (1)⟩]

theorem exact28609RawTermsValid :
    exact28609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28609 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17354⟩⟩) exact28609RawTerms (.finite 55) 28608 .exactZero (none)

def event28610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17361⟩⟩) 0 ⟨6544⟩ 28586

def event28611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17361⟩⟩) 1 ⟨17354⟩ 28609

def event28612 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17361⟩⟩) (.product (.predecessor 0 28610 .coefficient) (.predecessor 1 28611 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17361⟩⟩, .operator (⟨28586, 0⟩, ⟨28609, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28614RawTermsValid :
    exact28614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17361⟩⟩) exact28614RawTerms .large 28612 .exactZero (none)

def event28615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6715⟩⟩) 0 ⟨6689⟩ 28568

def event28616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6715⟩⟩) (.authority (.operator))

def exact28617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩]

theorem exact28617RawTermsValid :
    exact28617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6715⟩⟩) exact28617RawTerms .large 28616 .exactZero (none)

def event28618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17362⟩⟩) 0 ⟨6715⟩ 28617

def event28619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17362⟩⟩) 1 ⟨17361⟩ 28614

def event28620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17362⟩⟩) (.sum [.predecessor 0 28618 .coefficient, .predecessor 1 28619 .coefficient])

def exact28621RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28621RawTermsValid :
    exact28621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17362⟩⟩) exact28621RawTerms .large 28620 .exactZero (none)

def event28622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27042⟩⟩) 0 ⟨17362⟩ 28621

def event28623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27042⟩⟩) 1 ⟨27038⟩ 28606

def event28624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27042⟩⟩) (.sum [.predecessor 0 28622 .coefficient, .predecessor 1 28623 .coefficient])

def exact28625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28625RawTermsValid :
    exact28625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28625 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27042⟩⟩) exact28625RawTerms .large 28624 .exactZero (none)

def event28626 : Event := .preFoldPolynomial 28625 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event28627 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27042⟩⟩) 28626 exact28627RawTerms .large 28624 .exactZero (none)

def event28628 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15435⟩⟩) ⟨⟨128⟩, ⟨35⟩, ⟨109⟩⟩ ⟨28470, 28628⟩

def event28629 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20839⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩) (1) 0 2 (.universal 28628 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩) (none) 28627)

def event28630 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20839⟩⟩, .relation 28629 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩)

def event28631 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20839⟩⟩, .relation 28629 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩)

def event28632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20839⟩⟩, .relation 28629 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩)

def event28633 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20839⟩⟩, .relation 28629 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact28634RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28634RawTermsValid :
    exact28634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28634 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20839⟩⟩) exact28634RawTerms .large 28466 (.finite 1811303510016) (some (28468))

def event28635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27040⟩⟩) 0 ⟨20839⟩ 28634

def event28636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27040⟩⟩) 1 ⟨27039⟩ 28456

def event28637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27040⟩⟩) (.sum [.predecessor 0 28635 .coefficient, .predecessor 1 28636 .coefficient])

def event28638 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27040⟩⟩, .operator (⟨28634, 0⟩, ⟨28456, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩)

def event28639 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27040⟩⟩, .operator (⟨28634, 2⟩, ⟨28456, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15434⟩⟩], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (-1)⟩)

def event28640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27040⟩⟩) (.sum [.result 28634 .summary, .result 28456 .summary])

def exact28641RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28641RawTermsValid :
    exact28641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27040⟩⟩) exact28641RawTerms .large 28637 (.finite 1291933999269462814720) (some (28640))

def event28642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23854⟩⟩) 0 ⟨15127⟩ 1204

def event28643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23854⟩⟩) (.authority (.programFamilyFact))

def event28644 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23854⟩⟩) (.finite 3720)

def event28645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23856⟩⟩) 0 ⟨6689⟩ 5477

def event28646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23856⟩⟩) 1 ⟨23854⟩ 28644

def event28647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23856⟩⟩) (.authority (.operator))

def exact28648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23856⟩⟩]⟩, (1)⟩]

theorem exact28648RawTermsValid :
    exact28648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28648 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23856⟩⟩) exact28648RawTerms .large 28647 .exactZero (none)

def event28649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26820⟩⟩) 0 ⟨23856⟩ 28648

def event28650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26820⟩⟩) (.authority (.operator))

def exact28651RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩, (1)⟩]

theorem exact28651RawTermsValid :
    exact28651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28651 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26820⟩⟩) exact28651RawTerms (.finite 8192) 28650 .exactZero (none)

def event28652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23043⟩⟩) 0 ⟨11003⟩ 1198

def event28653 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23043⟩⟩) (.authority (.programFamilyFact))

def event28654 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23043⟩⟩) (.finite 3720)

def event28655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23044⟩⟩) 0 ⟨6689⟩ 5477

def event28656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23044⟩⟩) 1 ⟨23043⟩ 28654

def event28657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23044⟩⟩) (.authority (.operator))

def exact28658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23044⟩⟩]⟩, (1)⟩]

theorem exact28658RawTermsValid :
    exact28658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23044⟩⟩) exact28658RawTerms .large 28657 .exactZero (none)

def event28659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25080⟩⟩) 0 ⟨23044⟩ 28658

def event28660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25080⟩⟩) (.authority (.operator))

def exact28661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25080⟩⟩]⟩, (1)⟩]

theorem exact28661RawTermsValid :
    exact28661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28661 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25080⟩⟩) exact28661RawTerms (.finite 8192) 28660 .exactZero (none)

def event28662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11004⟩⟩) 0 ⟨11001⟩ 1187

def event28663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11004⟩⟩) 1 ⟨6570⟩ 21420

def event28664 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11004⟩⟩) (.tensor (.predecessor 0 28662 .coefficient) (.predecessor 1 28663 .coefficient) true false)

def event28665 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11004⟩⟩, .operator (⟨1187, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28666RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28666RawTermsValid :
    exact28666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11004⟩⟩) exact28666RawTerms .large 28664 .exactZero (none)

def event28667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7344⟩⟩) 0 ⟨5557⟩ 21290

def event28668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7344⟩⟩) 1 ⟨6774⟩ 13987

def event28669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7344⟩⟩) (.product (.predecessor 0 28667 .coefficient) (.predecessor 1 28668 .coefficient) (⟨false, false, none, none, none⟩))

def event28670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7344⟩⟩, .operator (⟨21290, 0⟩, ⟨13987, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩, (1)⟩)

def exact28671RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩, (1)⟩]

theorem exact28671RawTermsValid :
    exact28671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28671 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7344⟩⟩) exact28671RawTerms .large 28669 .exactZero (none)

def eventLeaf1776 : Array AnnotatedEvent := #[
  { event := event28416
    frameStart := 28315 },
  { event := event28417
    frameStart := 28315 },
  { event := event28418
    frameStart := 28315 },
  { event := event28419
    frameStart := 28315 },
  { event := event28420
    frameStart := 28315 },
  { event := event28421
    frameStart := 28315 },
  { event := event28422
    frameStart := 28315 },
  { event := event28423
    frameStart := 28315 },
  { event := event28424
    frameStart := 28315 },
  { event := event28425
    frameStart := 28315 },
  { event := event28426
    frameStart := 28315 },
  { event := event28427
    frameStart := 28315 },
  { event := event28428
    frameStart := 28315 },
  { event := event28429
    frameStart := 28315 },
  { event := event28430
    frameStart := 28315 },
  { event := event28431
    frameStart := 28315 }
]

def eventLeaf1777 : Array AnnotatedEvent := #[
  { event := event28432
    frameStart := 28315 },
  { event := event28433
    frameStart := 0 },
  { event := event28434
    frameStart := 0 },
  { event := event28435
    frameStart := 0 },
  { event := event28436
    frameStart := 0 },
  { event := event28437
    frameStart := 0 },
  { event := event28438
    frameStart := 0 },
  { event := event28439
    frameStart := 0 },
  { event := event28440
    frameStart := 0 },
  { event := event28441
    frameStart := 0 },
  { event := event28442
    frameStart := 0 },
  { event := event28443
    frameStart := 0 },
  { event := event28444
    frameStart := 0 },
  { event := event28445
    frameStart := 0 },
  { event := event28446
    frameStart := 0 },
  { event := event28447
    frameStart := 0 }
]

def eventLeaf1778 : Array AnnotatedEvent := #[
  { event := event28448
    frameStart := 0 },
  { event := event28449
    frameStart := 0 },
  { event := event28450
    frameStart := 0 },
  { event := event28451
    frameStart := 0 },
  { event := event28452
    frameStart := 0 },
  { event := event28453
    frameStart := 0 },
  { event := event28454
    frameStart := 0 },
  { event := event28455
    frameStart := 0 },
  { event := event28456
    frameStart := 0 },
  { event := event28457
    frameStart := 0 },
  { event := event28458
    frameStart := 0 },
  { event := event28459
    frameStart := 0 },
  { event := event28460
    frameStart := 0 },
  { event := event28461
    frameStart := 0 },
  { event := event28462
    frameStart := 0 },
  { event := event28463
    frameStart := 0 }
]

def eventLeaf1779 : Array AnnotatedEvent := #[
  { event := event28464
    frameStart := 0 },
  { event := event28465
    frameStart := 0 },
  { event := event28466
    frameStart := 0 },
  { event := event28467
    frameStart := 0 },
  { event := event28468
    frameStart := 0 },
  { event := event28469
    frameStart := 0 },
  { event := event28470
    frameStart := 28470 },
  { event := event28471
    frameStart := 28470 },
  { event := event28472
    frameStart := 28470 },
  { event := event28473
    frameStart := 28470 },
  { event := event28474
    frameStart := 28470 },
  { event := event28475
    frameStart := 28470 },
  { event := event28476
    frameStart := 28470 },
  { event := event28477
    frameStart := 28470 },
  { event := event28478
    frameStart := 28470 },
  { event := event28479
    frameStart := 28470 }
]

def eventLeaf1780 : Array AnnotatedEvent := #[
  { event := event28480
    frameStart := 28470 },
  { event := event28481
    frameStart := 28470 },
  { event := event28482
    frameStart := 28470 },
  { event := event28483
    frameStart := 28470 },
  { event := event28484
    frameStart := 28470 },
  { event := event28485
    frameStart := 28470 },
  { event := event28486
    frameStart := 28470 },
  { event := event28487
    frameStart := 28470 },
  { event := event28488
    frameStart := 28470 },
  { event := event28489
    frameStart := 28470 },
  { event := event28490
    frameStart := 28470 },
  { event := event28491
    frameStart := 28470 },
  { event := event28492
    frameStart := 28470 },
  { event := event28493
    frameStart := 28470 },
  { event := event28494
    frameStart := 28470 },
  { event := event28495
    frameStart := 28470 }
]

def eventLeaf1781 : Array AnnotatedEvent := #[
  { event := event28496
    frameStart := 28470 },
  { event := event28497
    frameStart := 28470 },
  { event := event28498
    frameStart := 28470 },
  { event := event28499
    frameStart := 28470 },
  { event := event28500
    frameStart := 28470 },
  { event := event28501
    frameStart := 28470 },
  { event := event28502
    frameStart := 28470 },
  { event := event28503
    frameStart := 28470 },
  { event := event28504
    frameStart := 28470 },
  { event := event28505
    frameStart := 28470 },
  { event := event28506
    frameStart := 28470 },
  { event := event28507
    frameStart := 28470 },
  { event := event28508
    frameStart := 28470 },
  { event := event28509
    frameStart := 28470 },
  { event := event28510
    frameStart := 28470 },
  { event := event28511
    frameStart := 28470 }
]

def eventLeaf1782 : Array AnnotatedEvent := #[
  { event := event28512
    frameStart := 28470 },
  { event := event28513
    frameStart := 28470 },
  { event := event28514
    frameStart := 28470 },
  { event := event28515
    frameStart := 28470 },
  { event := event28516
    frameStart := 28470 },
  { event := event28517
    frameStart := 28470 },
  { event := event28518
    frameStart := 28470 },
  { event := event28519
    frameStart := 28470 },
  { event := event28520
    frameStart := 28470 },
  { event := event28521
    frameStart := 28470 },
  { event := event28522
    frameStart := 28470 },
  { event := event28523
    frameStart := 28470 },
  { event := event28524
    frameStart := 28524 },
  { event := event28525
    frameStart := 28524 },
  { event := event28526
    frameStart := 28524 },
  { event := event28527
    frameStart := 28524 }
]

def eventLeaf1783 : Array AnnotatedEvent := #[
  { event := event28528
    frameStart := 28524 },
  { event := event28529
    frameStart := 28524 },
  { event := event28530
    frameStart := 28524 },
  { event := event28531
    frameStart := 28524 },
  { event := event28532
    frameStart := 28524 },
  { event := event28533
    frameStart := 28524 },
  { event := event28534
    frameStart := 28524 },
  { event := event28535
    frameStart := 28524 },
  { event := event28536
    frameStart := 28524 },
  { event := event28537
    frameStart := 28524 },
  { event := event28538
    frameStart := 28524 },
  { event := event28539
    frameStart := 28524 },
  { event := event28540
    frameStart := 28524 },
  { event := event28541
    frameStart := 28524 },
  { event := event28542
    frameStart := 28524 },
  { event := event28543
    frameStart := 28524 }
]

def eventLeaf1784 : Array AnnotatedEvent := #[
  { event := event28544
    frameStart := 28524 },
  { event := event28545
    frameStart := 28524 },
  { event := event28546
    frameStart := 28524 },
  { event := event28547
    frameStart := 28524 },
  { event := event28548
    frameStart := 28524 },
  { event := event28549
    frameStart := 28524 },
  { event := event28550
    frameStart := 28524 },
  { event := event28551
    frameStart := 28524 },
  { event := event28552
    frameStart := 28524 },
  { event := event28553
    frameStart := 28524 },
  { event := event28554
    frameStart := 28524 },
  { event := event28555
    frameStart := 28524 },
  { event := event28556
    frameStart := 28524 },
  { event := event28557
    frameStart := 28524 },
  { event := event28558
    frameStart := 28524 },
  { event := event28559
    frameStart := 28524 }
]

def eventLeaf1785 : Array AnnotatedEvent := #[
  { event := event28560
    frameStart := 28524 },
  { event := event28561
    frameStart := 28524 },
  { event := event28562
    frameStart := 28524 },
  { event := event28563
    frameStart := 28524 },
  { event := event28564
    frameStart := 28524 },
  { event := event28565
    frameStart := 28524 },
  { event := event28566
    frameStart := 28524 },
  { event := event28567
    frameStart := 28524 },
  { event := event28568
    frameStart := 28524 },
  { event := event28569
    frameStart := 28524 },
  { event := event28570
    frameStart := 28524 },
  { event := event28571
    frameStart := 28524 },
  { event := event28572
    frameStart := 28524 },
  { event := event28573
    frameStart := 28524 },
  { event := event28574
    frameStart := 28524 },
  { event := event28575
    frameStart := 28524 }
]

def eventLeaf1786 : Array AnnotatedEvent := #[
  { event := event28576
    frameStart := 28524 },
  { event := event28577
    frameStart := 28524 },
  { event := event28578
    frameStart := 28524 },
  { event := event28579
    frameStart := 28524 },
  { event := event28580
    frameStart := 28524 },
  { event := event28581
    frameStart := 28524 },
  { event := event28582
    frameStart := 28524 },
  { event := event28583
    frameStart := 28524 },
  { event := event28584
    frameStart := 28524 },
  { event := event28585
    frameStart := 28524 },
  { event := event28586
    frameStart := 28524 },
  { event := event28587
    frameStart := 28524 },
  { event := event28588
    frameStart := 28524 },
  { event := event28589
    frameStart := 28524 },
  { event := event28590
    frameStart := 28524 },
  { event := event28591
    frameStart := 28524 }
]

def eventLeaf1787 : Array AnnotatedEvent := #[
  { event := event28592
    frameStart := 28524 },
  { event := event28593
    frameStart := 28524 },
  { event := event28594
    frameStart := 28524 },
  { event := event28595
    frameStart := 28524 },
  { event := event28596
    frameStart := 28524 },
  { event := event28597
    frameStart := 28524 },
  { event := event28598
    frameStart := 28524 },
  { event := event28599
    frameStart := 28524 },
  { event := event28600
    frameStart := 28524 },
  { event := event28601
    frameStart := 28524 },
  { event := event28602
    frameStart := 28524 },
  { event := event28603
    frameStart := 28524 },
  { event := event28604
    frameStart := 28524 },
  { event := event28605
    frameStart := 28524 },
  { event := event28606
    frameStart := 28524 },
  { event := event28607
    frameStart := 28524 }
]

def eventLeaf1788 : Array AnnotatedEvent := #[
  { event := event28608
    frameStart := 28524 },
  { event := event28609
    frameStart := 28524 },
  { event := event28610
    frameStart := 28524 },
  { event := event28611
    frameStart := 28524 },
  { event := event28612
    frameStart := 28524 },
  { event := event28613
    frameStart := 28524 },
  { event := event28614
    frameStart := 28524 },
  { event := event28615
    frameStart := 28524 },
  { event := event28616
    frameStart := 28524 },
  { event := event28617
    frameStart := 28524 },
  { event := event28618
    frameStart := 28524 },
  { event := event28619
    frameStart := 28524 },
  { event := event28620
    frameStart := 28524 },
  { event := event28621
    frameStart := 28524 },
  { event := event28622
    frameStart := 28524 },
  { event := event28623
    frameStart := 28524 }
]

def eventLeaf1789 : Array AnnotatedEvent := #[
  { event := event28624
    frameStart := 28524 },
  { event := event28625
    frameStart := 28524 },
  { event := event28626
    frameStart := 28524 },
  { event := event28627
    frameStart := 28524 },
  { event := event28628
    frameStart := 0 },
  { event := event28629
    frameStart := 0 },
  { event := event28630
    frameStart := 0 },
  { event := event28631
    frameStart := 0 },
  { event := event28632
    frameStart := 0 },
  { event := event28633
    frameStart := 0 },
  { event := event28634
    frameStart := 0 },
  { event := event28635
    frameStart := 0 },
  { event := event28636
    frameStart := 0 },
  { event := event28637
    frameStart := 0 },
  { event := event28638
    frameStart := 0 },
  { event := event28639
    frameStart := 0 }
]

def eventLeaf1790 : Array AnnotatedEvent := #[
  { event := event28640
    frameStart := 0 },
  { event := event28641
    frameStart := 0 },
  { event := event28642
    frameStart := 0 },
  { event := event28643
    frameStart := 0 },
  { event := event28644
    frameStart := 0 },
  { event := event28645
    frameStart := 0 },
  { event := event28646
    frameStart := 0 },
  { event := event28647
    frameStart := 0 },
  { event := event28648
    frameStart := 0 },
  { event := event28649
    frameStart := 0 },
  { event := event28650
    frameStart := 0 },
  { event := event28651
    frameStart := 0 },
  { event := event28652
    frameStart := 0 },
  { event := event28653
    frameStart := 0 },
  { event := event28654
    frameStart := 0 },
  { event := event28655
    frameStart := 0 }
]

def eventLeaf1791 : Array AnnotatedEvent := #[
  { event := event28656
    frameStart := 0 },
  { event := event28657
    frameStart := 0 },
  { event := event28658
    frameStart := 0 },
  { event := event28659
    frameStart := 0 },
  { event := event28660
    frameStart := 0 },
  { event := event28661
    frameStart := 0 },
  { event := event28662
    frameStart := 0 },
  { event := event28663
    frameStart := 0 },
  { event := event28664
    frameStart := 0 },
  { event := event28665
    frameStart := 0 },
  { event := event28666
    frameStart := 0 },
  { event := event28667
    frameStart := 0 },
  { event := event28668
    frameStart := 0 },
  { event := event28669
    frameStart := 0 },
  { event := event28670
    frameStart := 0 },
  { event := event28671
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events111
