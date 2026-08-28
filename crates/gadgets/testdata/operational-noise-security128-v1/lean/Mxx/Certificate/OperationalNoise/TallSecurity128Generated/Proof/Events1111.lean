import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1111

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event284416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28139⟩⟩) 0 ⟨27507⟩ 284415

def event284417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28139⟩⟩) (.authority (.operator))

def exact284418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩]

theorem exact284418RawTermsValid :
    exact284418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28139⟩⟩) exact284418RawTerms (.finite 8192) 284417 .exactZero (none)

def event284419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event284420 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event284421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27742⟩⟩) 0 ⟨26361⟩ 284407

def event284422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27742⟩⟩) 1 ⟨136⟩ 284420

def event284423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27742⟩⟩) (.sum [.predecessor 0 284421 .coefficient, .predecessor 1 284422 .coefficient])

def event284424 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27742⟩⟩) (.finite 30)

def event284425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27743⟩⟩) 0 ⟨27742⟩ 284424

def event284426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27743⟩⟩) (.identity (.predecessor 0 284425 .coefficient))

def exact284427RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], []⟩, (1)⟩]

theorem exact284427RawTermsValid :
    exact284427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27743⟩⟩) exact284427RawTerms (.finite 30) 284426 .exactZero (none)

def event284428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact284429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284429RawTermsValid :
    exact284429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284429 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact284429RawTerms .large 284428 .exactZero (none)

def event284430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27744⟩⟩) 0 ⟨6908⟩ 284429

def event284431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27744⟩⟩) 1 ⟨27743⟩ 284427

def event284432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27744⟩⟩) (.product (.predecessor 0 284430 .coefficient) (.predecessor 1 284431 .coefficient) (⟨false, false, none, none, none⟩))

def event284433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27744⟩⟩, .operator (⟨284429, 0⟩, ⟨284427, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284434RawTermsValid :
    exact284434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27744⟩⟩) exact284434RawTerms .large 284432 .exactZero (none)

def event284435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 284411

def event284436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact284437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact284437RawTermsValid :
    exact284437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284437 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact284437RawTerms .large 284436 .exactZero (none)

def event284438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27745⟩⟩) 0 ⟨7189⟩ 284437

def event284439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27745⟩⟩) 1 ⟨27744⟩ 284434

def event284440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27745⟩⟩) (.sum [.predecessor 0 284438 .coefficient, .predecessor 1 284439 .coefficient])

def exact284441RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284441RawTermsValid :
    exact284441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284441 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27745⟩⟩) exact284441RawTerms .large 284440 .exactZero (none)

def event284442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28140⟩⟩) 0 ⟨27745⟩ 284441

def event284443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28140⟩⟩) 1 ⟨28139⟩ 284418

def event284444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28140⟩⟩) (.product (.predecessor 0 284442 .coefficient) (.predecessor 1 284443 .coefficient) (⟨false, false, none, none, none⟩))

def event284445 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28140⟩⟩, .operator (⟨284441, 0⟩, ⟨284418, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩)

def event284446 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28140⟩⟩, .operator (⟨284441, 1⟩, ⟨284418, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩)

def event284447 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28140⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28139⟩⟩) ⟨27507⟩ 284415)

def event284448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28140⟩⟩, .relation 284447 0, ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (-1)⟩)

def exact284449RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (-1)⟩]

theorem exact284449RawTermsValid :
    exact284449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28140⟩⟩) exact284449RawTerms .large 284444 .exactZero (none)

def event284450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26541⟩⟩) 0 ⟨26361⟩ 284407

def event284451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26541⟩⟩) (.authority (.programFamilyFact))

def exact284452RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], []⟩, (1)⟩]

theorem exact284452RawTermsValid :
    exact284452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26541⟩⟩) exact284452RawTerms (.finite 62) 284451 .exactZero (none)

def event284453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26542⟩⟩) 0 ⟨6908⟩ 284429

def event284454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26542⟩⟩) 1 ⟨26541⟩ 284452

def event284455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26542⟩⟩) (.product (.predecessor 0 284453 .coefficient) (.predecessor 1 284454 .coefficient) (⟨false, true, none, none, some 1⟩))

def event284456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26542⟩⟩, .operator (⟨284429, 0⟩, ⟨284452, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284457RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284457RawTermsValid :
    exact284457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26542⟩⟩) exact284457RawTerms .large 284455 .exactZero (none)

def event284458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 284411

def event284459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact284460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact284460RawTermsValid :
    exact284460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact284460RawTerms .large 284459 .exactZero (none)

def event284461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26543⟩⟩) 0 ⟨7218⟩ 284460

def event284462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26543⟩⟩) 1 ⟨26542⟩ 284457

def event284463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26543⟩⟩) (.sum [.predecessor 0 284461 .coefficient, .predecessor 1 284462 .coefficient])

def exact284464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284464RawTermsValid :
    exact284464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26543⟩⟩) exact284464RawTerms .large 284463 .exactZero (none)

def event284465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28143⟩⟩) 0 ⟨26543⟩ 284464

def event284466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28143⟩⟩) 1 ⟨28140⟩ 284449

def event284467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28143⟩⟩) (.sum [.predecessor 0 284465 .coefficient, .predecessor 1 284466 .coefficient])

def exact284468RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284468RawTermsValid :
    exact284468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28143⟩⟩) exact284468RawTerms .large 284467 .exactZero (none)

def event284469 : Event := .preFoldPolynomial 284468 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact284470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event284470 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28143⟩⟩) 284469 exact284470RawTerms .large 284467 .exactZero (none)

def event284471 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26361⟩⟩) ⟨⟨97⟩, ⟨79⟩, ⟨135⟩⟩ ⟨284313, 284471⟩

def event284472 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27039⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩) (1) 0 2 (.universal 284471 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩) (none) 284470)

def event284473 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27039⟩⟩, .relation 284472 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩)

def event284474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27039⟩⟩, .relation 284472 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩)

def event284475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27039⟩⟩, .relation 284472 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩)

def event284476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27039⟩⟩, .relation 284472 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact284477RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284477RawTermsValid :
    exact284477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284477 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27039⟩⟩) exact284477RawTerms .large 284309 (.finite 202072841853861888) (some (284311))

def event284478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28142⟩⟩) 0 ⟨27039⟩ 284477

def event284479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28142⟩⟩) 1 ⟨28141⟩ 284299

def event284480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28142⟩⟩) (.sum [.predecessor 0 284478 .coefficient, .predecessor 1 284479 .coefficient])

def event284481 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28142⟩⟩, .operator (⟨284477, 0⟩, ⟨284299, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩)

def event284482 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28142⟩⟩, .operator (⟨284477, 2⟩, ⟨284299, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (-1)⟩)

def event284483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28142⟩⟩) (.sum [.result 284477 .summary, .result 284299 .summary])

def exact284484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284484RawTermsValid :
    exact284484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28142⟩⟩) exact284484RawTerms .large 284480 (.finite 32191557518723330170883082027008) (some (284483))

def event284485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68626⟩⟩) 0 ⟨65741⟩ 13753

def event284486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68626⟩⟩) (.authority (.programFamilyFact))

def event284487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68626⟩⟩) (.finite 3720)

def event284488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68628⟩⟩) 0 ⟨7177⟩ 15500

def event284489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68628⟩⟩) 1 ⟨68626⟩ 284487

def event284490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68628⟩⟩) (.authority (.operator))

def exact284491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩, (1)⟩]

theorem exact284491RawTermsValid :
    exact284491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284491 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68628⟩⟩) exact284491RawTerms .large 284490 .exactZero (none)

def event284492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69703⟩⟩) 0 ⟨68628⟩ 284491

def event284493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69703⟩⟩) (.authority (.operator))

def exact284494RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (1)⟩]

theorem exact284494RawTermsValid :
    exact284494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284494 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69703⟩⟩) exact284494RawTerms (.finite 8192) 284493 .exactZero (none)

def event284495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68493⟩⟩) 0 ⟨65285⟩ 13747

def event284496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68493⟩⟩) (.authority (.programFamilyFact))

def event284497 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68493⟩⟩) (.finite 3720)

def event284498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68494⟩⟩) 0 ⟨7177⟩ 15500

def event284499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68494⟩⟩) 1 ⟨68493⟩ 284497

def event284500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68494⟩⟩) (.authority (.operator))

def exact284501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩]

theorem exact284501RawTermsValid :
    exact284501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284501 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68494⟩⟩) exact284501RawTerms .large 284500 .exactZero (none)

def event284502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69174⟩⟩) 0 ⟨68494⟩ 284501

def event284503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69174⟩⟩) (.authority (.operator))

def exact284504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩]

theorem exact284504RawTermsValid :
    exact284504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69174⟩⟩) exact284504RawTerms (.finite 8192) 284503 .exactZero (none)

def event284505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25659⟩⟩) 0 ⟨25658⟩ 13736

def event284506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25659⟩⟩) 1 ⟨6922⟩ 280653

def event284507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25659⟩⟩) (.tensor (.predecessor 0 284505 .coefficient) (.predecessor 1 284506 .coefficient) true false)

def event284508 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25659⟩⟩, .operator (⟨13736, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284509RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284509RawTermsValid :
    exact284509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284509 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25659⟩⟩) exact284509RawTerms .large 284507 .exactZero (none)

def event284510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7898⟩⟩) 0 ⟨5489⟩ 280523

def event284511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7898⟩⟩) 1 ⟨7276⟩ 21088

def event284512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7898⟩⟩) (.product (.predecessor 0 284510 .coefficient) (.predecessor 1 284511 .coefficient) (⟨false, false, none, none, none⟩))

def event284513 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7898⟩⟩, .operator (⟨280523, 0⟩, ⟨21088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact284514RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact284514RawTermsValid :
    exact284514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7898⟩⟩) exact284514RawTerms .large 284512 .exactZero (none)

def event284515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25660⟩⟩) 0 ⟨7898⟩ 284514

def event284516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25660⟩⟩) 1 ⟨25659⟩ 284509

def event284517 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25660⟩⟩) (.sum [.predecessor 0 284515 .coefficient, .predecessor 1 284516 .coefficient])

def exact284518RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284518RawTermsValid :
    exact284518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284518 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25660⟩⟩) exact284518RawTerms .large 284517 .exactZero (none)

def event284519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25661⟩⟩) 0 ⟨25660⟩ 284518

def event284520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25661⟩⟩) 1 ⟨102⟩ 21080

def event284521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25661⟩⟩) (.sum [.predecessor 0 284519 .coefficient, .predecessor 1 284520 .coefficient])

def event284522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25661⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩) [⟨.result 21080 .coefficient, false, none⟩])

def event284523 : Event := .survivorFold (1) 284522

def exact284524RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284524RawTermsValid :
    exact284524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25661⟩⟩) exact284524RawTerms .large 284521 (.finite 26) (some (284522))

def event284525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65286⟩⟩) 0 ⟨25661⟩ 284524

def event284526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65286⟩⟩) 1 ⟨65283⟩ 13739

def event284527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65286⟩⟩) (.product (.predecessor 0 284525 .coefficient) (.predecessor 1 284526 .coefficient) (⟨false, true, none, none, some 1⟩))

def event284528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65286⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩) [⟨.result 13739 .coefficient, true, some 1⟩])

def event284529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65286⟩⟩) (.product (.result 284524 .summary) (.transfer 284528) (⟨false, false, none, none, none⟩))

def event284530 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65286⟩⟩, .operator (⟨284524, 1⟩, ⟨13739, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event284531 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65286⟩⟩, .operator (⟨284524, 0⟩, ⟨13739, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact284532RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact284532RawTermsValid :
    exact284532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65286⟩⟩) exact284532RawTerms .large 284527 (.finite 23855104) (some (284529))

def event284533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65287⟩⟩) 0 ⟨65283⟩ 13739

def event284534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65287⟩⟩) 1 ⟨6922⟩ 280653

def event284535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65287⟩⟩) (.tensor (.predecessor 0 284533 .coefficient) (.predecessor 1 284534 .coefficient) true false)

def event284536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65287⟩⟩, .operator (⟨13739, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284537RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284537RawTermsValid :
    exact284537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65287⟩⟩) exact284537RawTerms .large 284535 .exactZero (none)

def event284538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7916⟩⟩) 0 ⟨5489⟩ 280523

def event284539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7916⟩⟩) 1 ⟨7294⟩ 21129

def event284540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7916⟩⟩) (.product (.predecessor 0 284538 .coefficient) (.predecessor 1 284539 .coefficient) (⟨false, false, none, none, none⟩))

def event284541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7916⟩⟩, .operator (⟨280523, 0⟩, ⟨21129, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩)

def exact284542RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact284542RawTermsValid :
    exact284542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284542 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7916⟩⟩) exact284542RawTerms .large 284540 .exactZero (none)

def event284543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65288⟩⟩) 0 ⟨7916⟩ 284542

def event284544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65288⟩⟩) 1 ⟨65287⟩ 284537

def event284545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65288⟩⟩) (.sum [.predecessor 0 284543 .coefficient, .predecessor 1 284544 .coefficient])

def exact284546RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284546RawTermsValid :
    exact284546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284546 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65288⟩⟩) exact284546RawTerms .large 284545 .exactZero (none)

def event284547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65289⟩⟩) 0 ⟨65288⟩ 284546

def event284548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65289⟩⟩) 1 ⟨120⟩ 21121

def event284549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65289⟩⟩) (.sum [.predecessor 0 284547 .coefficient, .predecessor 1 284548 .coefficient])

def event284550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65289⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨120⟩⟩]⟩) [⟨.result 21121 .coefficient, false, none⟩])

def event284551 : Event := .survivorFold (1) 284550

def exact284552RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284552RawTermsValid :
    exact284552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65289⟩⟩) exact284552RawTerms .large 284549 (.finite 26) (some (284550))

def event284553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65290⟩⟩) 0 ⟨65289⟩ 284552

def event284554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65290⟩⟩) 1 ⟨9542⟩ 21118

def event284555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65290⟩⟩) (.product (.predecessor 0 284553 .coefficient) (.predecessor 1 284554 .coefficient) (⟨false, false, none, none, none⟩))

def event284556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65290⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) [⟨.result 21114 .coefficient, false, none⟩])

def event284557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65290⟩⟩) (.product (.result 284552 .summary) (.transfer 284556) (⟨false, false, none, none, none⟩))

def event284558 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65290⟩⟩, .operator (⟨284552, 1⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (-1)⟩)

def event284559 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨65290⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088)

def event284560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65290⟩⟩, .relation 284559 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩)

def event284561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65290⟩⟩, .operator (⟨284552, 0⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact284562RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩]

theorem exact284562RawTermsValid :
    exact284562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65290⟩⟩) exact284562RawTerms .large 284555 (.finite 279172874240) (some (284557))

def event284563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65291⟩⟩) 0 ⟨65290⟩ 284562

def event284564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65291⟩⟩) 1 ⟨65286⟩ 284532

def event284565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65291⟩⟩) (.sum [.predecessor 0 284563 .coefficient, .predecessor 1 284564 .coefficient])

def event284566 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65291⟩⟩, .operator (⟨284562, 1⟩, ⟨284532, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def event284567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65291⟩⟩) (.sum [.result 284562 .summary, .result 284532 .summary])

def exact284568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284568RawTermsValid :
    exact284568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65291⟩⟩) exact284568RawTerms .large 284565 (.finite 279196729344) (some (284567))

def event284569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69175⟩⟩) 0 ⟨65291⟩ 284568

def event284570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69175⟩⟩) 1 ⟨69174⟩ 284504

def event284571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69175⟩⟩) (.product (.predecessor 0 284569 .coefficient) (.predecessor 1 284570 .coefficient) (⟨false, false, none, none, none⟩))

def event284572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69175⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩) [⟨.result 284504 .coefficient, false, none⟩])

def event284573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69175⟩⟩) (.product (.result 284568 .summary) (.transfer 284572) (⟨false, false, none, none, none⟩))

def event284574 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69175⟩⟩, .operator (⟨284568, 1⟩, ⟨284504, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩)

def event284575 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69175⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69174⟩⟩) ⟨68494⟩ 284501)

def event284576 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69175⟩⟩, .relation 284575 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (-1)⟩)

def event284577 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69175⟩⟩, .operator (⟨284568, 0⟩, ⟨284504, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩)

def exact284578RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (-1)⟩]

theorem exact284578RawTermsValid :
    exact284578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284578 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69175⟩⟩) exact284578RawTerms .large 284571 (.finite 2997852054206608834560) (some (284573))

def event284579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67710⟩⟩) 0 ⟨65285⟩ 13747

def event284580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67710⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact284581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩]

theorem exact284581RawTermsValid :
    exact284581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67710⟩⟩) exact284581RawTerms (.finite 5647228698) 284580 .exactZero (none)

def event284582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67712⟩⟩) 0 ⟨67710⟩ 284581

def event284583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67712⟩⟩) 1 ⟨2370⟩ 4

def event284584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67712⟩⟩) (.scale (.predecessor 0 284582 .coefficient) (.value (.predecessor 1 284583 .coefficient)))

def exact284585RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩]

theorem exact284585RawTermsValid :
    exact284585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284585 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67712⟩⟩) exact284585RawTerms (.finite 5647228698) 284584 .exactZero (none)

def event284586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67713⟩⟩) 0 ⟨5491⟩ 280745

def event284587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67713⟩⟩) 1 ⟨67712⟩ 284585

def event284588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67713⟩⟩) (.product (.predecessor 0 284586 .coefficient) (.predecessor 1 284587 .coefficient) (⟨false, false, none, none, none⟩))

def event284589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67713⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩) [⟨.result 284581 .coefficient, false, none⟩])

def event284590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67713⟩⟩) (.product (.result 280745 .summary) (.transfer 284589) (⟨false, false, none, none, none⟩))

def event284591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67713⟩⟩, .operator (⟨280745, 0⟩, ⟨284585, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩)

def event284592 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨67711⟩⟩)

def event284593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284594 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284596 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284598 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284600

def event284602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284598

def event284603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284601 .coefficient) (.value (.predecessor 1 284602 .coefficient)))

def event284604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284604

def event284606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284596

def event284607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284605 .coefficient, .predecessor 1 284606 .coefficient])

def event284608 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284608

def event284610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284594

def event284611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284610 .coefficient))

def event284612 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25658⟩⟩) 0 ⟨5487⟩ 284612

def event284614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25658⟩⟩) (.authority (.programFamilyFact))

def exact284615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩], []⟩, (1)⟩]

theorem exact284615RawTermsValid :
    exact284615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25658⟩⟩) exact284615RawTerms (.finite 28) 284614 .exactZero (none)

def event284616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65283⟩⟩) 0 ⟨5487⟩ 284612

def event284617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65283⟩⟩) (.authority (.programFamilyFact))

def exact284618RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284618RawTermsValid :
    exact284618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65283⟩⟩) exact284618RawTerms (.finite 28) 284617 .exactZero (none)

def event284619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 0 ⟨65283⟩ 284618

def event284620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 1 ⟨25658⟩ 284615

def event284621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.product (.predecessor 0 284619 .coefficient) (.predecessor 1 284620 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩) [⟨.result 284618 .coefficient, true, some 1⟩, ⟨.result 284615 .coefficient, true, some 1⟩])

def event284623 : Event := .survivorFold (1) 284622

def exact284624RawTerms : List Term := []

theorem exact284624RawTermsValid :
    exact284624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65284⟩⟩) exact284624RawTerms (.finite 784) 284621 (.finite 784) (some (284622))

def event284625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65285⟩⟩) 0 ⟨65284⟩ 284624

def event284626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.identity (.predecessor 0 284625 .coefficient))

def event284627 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.finite 784)

def event284628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67710⟩⟩) 0 ⟨65285⟩ 284627

def event284629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67710⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact284630RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩]

theorem exact284630RawTermsValid :
    exact284630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67710⟩⟩) exact284630RawTerms (.finite 5647228698) 284629 .exactZero (none)

def event284631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact284632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact284632RawTermsValid :
    exact284632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact284632RawTerms .large 284631 .exactZero (none)

def event284633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67711⟩⟩) 0 ⟨35⟩ 284632

def event284634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67711⟩⟩) 1 ⟨67710⟩ 284630

def event284635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67711⟩⟩) (.product (.predecessor 0 284633 .coefficient) (.predecessor 1 284634 .coefficient) (⟨false, false, none, none, none⟩))

def event284636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67711⟩⟩, .operator (⟨284632, 0⟩, ⟨284630, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩)

def exact284637RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩]

theorem exact284637RawTermsValid :
    exact284637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284637 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67711⟩⟩) exact284637RawTerms .large 284635 .exactZero (none)

def event284638 : Event := .preFoldPolynomial 284637 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩] .exactZero none

def exact284639RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩, (1)⟩]

def event284639 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨67711⟩⟩) 284638 exact284639RawTerms .large 284635 .exactZero (none)

def event284640 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨69178⟩⟩)

def event284641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284642 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284644 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284646 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284648 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284648

def event284650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284646

def event284651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284649 .coefficient) (.value (.predecessor 1 284650 .coefficient)))

def event284652 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284652

def event284654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284644

def event284655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284653 .coefficient, .predecessor 1 284654 .coefficient])

def event284656 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284656

def event284658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284642

def event284659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284658 .coefficient))

def event284660 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25658⟩⟩) 0 ⟨5487⟩ 284660

def event284662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25658⟩⟩) (.authority (.programFamilyFact))

def exact284663RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩], []⟩, (1)⟩]

theorem exact284663RawTermsValid :
    exact284663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25658⟩⟩) exact284663RawTerms (.finite 28) 284662 .exactZero (none)

def event284664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65283⟩⟩) 0 ⟨5487⟩ 284660

def event284665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65283⟩⟩) (.authority (.programFamilyFact))

def exact284666RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284666RawTermsValid :
    exact284666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65283⟩⟩) exact284666RawTerms (.finite 28) 284665 .exactZero (none)

def event284667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 0 ⟨65283⟩ 284666

def event284668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 1 ⟨25658⟩ 284663

def event284669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.product (.predecessor 0 284667 .coefficient) (.predecessor 1 284668 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65284⟩⟩, .operator (⟨284666, 0⟩, ⟨284663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩)

def exact284671RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284671RawTermsValid :
    exact284671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284671 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65284⟩⟩) exact284671RawTerms (.finite 784) 284669 .exactZero (none)

def eventLeaf17776 : Array AnnotatedEvent := #[
  { event := event284416
    frameStart := 284367 },
  { event := event284417
    frameStart := 284367 },
  { event := event284418
    frameStart := 284367 },
  { event := event284419
    frameStart := 284367 },
  { event := event284420
    frameStart := 284367 },
  { event := event284421
    frameStart := 284367 },
  { event := event284422
    frameStart := 284367 },
  { event := event284423
    frameStart := 284367 },
  { event := event284424
    frameStart := 284367 },
  { event := event284425
    frameStart := 284367 },
  { event := event284426
    frameStart := 284367 },
  { event := event284427
    frameStart := 284367 },
  { event := event284428
    frameStart := 284367 },
  { event := event284429
    frameStart := 284367 },
  { event := event284430
    frameStart := 284367 },
  { event := event284431
    frameStart := 284367 }
]

def eventLeaf17777 : Array AnnotatedEvent := #[
  { event := event284432
    frameStart := 284367 },
  { event := event284433
    frameStart := 284367 },
  { event := event284434
    frameStart := 284367 },
  { event := event284435
    frameStart := 284367 },
  { event := event284436
    frameStart := 284367 },
  { event := event284437
    frameStart := 284367 },
  { event := event284438
    frameStart := 284367 },
  { event := event284439
    frameStart := 284367 },
  { event := event284440
    frameStart := 284367 },
  { event := event284441
    frameStart := 284367 },
  { event := event284442
    frameStart := 284367 },
  { event := event284443
    frameStart := 284367 },
  { event := event284444
    frameStart := 284367 },
  { event := event284445
    frameStart := 284367 },
  { event := event284446
    frameStart := 284367 },
  { event := event284447
    frameStart := 284367 }
]

def eventLeaf17778 : Array AnnotatedEvent := #[
  { event := event284448
    frameStart := 284367 },
  { event := event284449
    frameStart := 284367 },
  { event := event284450
    frameStart := 284367 },
  { event := event284451
    frameStart := 284367 },
  { event := event284452
    frameStart := 284367 },
  { event := event284453
    frameStart := 284367 },
  { event := event284454
    frameStart := 284367 },
  { event := event284455
    frameStart := 284367 },
  { event := event284456
    frameStart := 284367 },
  { event := event284457
    frameStart := 284367 },
  { event := event284458
    frameStart := 284367 },
  { event := event284459
    frameStart := 284367 },
  { event := event284460
    frameStart := 284367 },
  { event := event284461
    frameStart := 284367 },
  { event := event284462
    frameStart := 284367 },
  { event := event284463
    frameStart := 284367 }
]

def eventLeaf17779 : Array AnnotatedEvent := #[
  { event := event284464
    frameStart := 284367 },
  { event := event284465
    frameStart := 284367 },
  { event := event284466
    frameStart := 284367 },
  { event := event284467
    frameStart := 284367 },
  { event := event284468
    frameStart := 284367 },
  { event := event284469
    frameStart := 284367 },
  { event := event284470
    frameStart := 284367 },
  { event := event284471
    frameStart := 0 },
  { event := event284472
    frameStart := 0 },
  { event := event284473
    frameStart := 0 },
  { event := event284474
    frameStart := 0 },
  { event := event284475
    frameStart := 0 },
  { event := event284476
    frameStart := 0 },
  { event := event284477
    frameStart := 0 },
  { event := event284478
    frameStart := 0 },
  { event := event284479
    frameStart := 0 }
]

def eventLeaf17780 : Array AnnotatedEvent := #[
  { event := event284480
    frameStart := 0 },
  { event := event284481
    frameStart := 0 },
  { event := event284482
    frameStart := 0 },
  { event := event284483
    frameStart := 0 },
  { event := event284484
    frameStart := 0 },
  { event := event284485
    frameStart := 0 },
  { event := event284486
    frameStart := 0 },
  { event := event284487
    frameStart := 0 },
  { event := event284488
    frameStart := 0 },
  { event := event284489
    frameStart := 0 },
  { event := event284490
    frameStart := 0 },
  { event := event284491
    frameStart := 0 },
  { event := event284492
    frameStart := 0 },
  { event := event284493
    frameStart := 0 },
  { event := event284494
    frameStart := 0 },
  { event := event284495
    frameStart := 0 }
]

def eventLeaf17781 : Array AnnotatedEvent := #[
  { event := event284496
    frameStart := 0 },
  { event := event284497
    frameStart := 0 },
  { event := event284498
    frameStart := 0 },
  { event := event284499
    frameStart := 0 },
  { event := event284500
    frameStart := 0 },
  { event := event284501
    frameStart := 0 },
  { event := event284502
    frameStart := 0 },
  { event := event284503
    frameStart := 0 },
  { event := event284504
    frameStart := 0 },
  { event := event284505
    frameStart := 0 },
  { event := event284506
    frameStart := 0 },
  { event := event284507
    frameStart := 0 },
  { event := event284508
    frameStart := 0 },
  { event := event284509
    frameStart := 0 },
  { event := event284510
    frameStart := 0 },
  { event := event284511
    frameStart := 0 }
]

def eventLeaf17782 : Array AnnotatedEvent := #[
  { event := event284512
    frameStart := 0 },
  { event := event284513
    frameStart := 0 },
  { event := event284514
    frameStart := 0 },
  { event := event284515
    frameStart := 0 },
  { event := event284516
    frameStart := 0 },
  { event := event284517
    frameStart := 0 },
  { event := event284518
    frameStart := 0 },
  { event := event284519
    frameStart := 0 },
  { event := event284520
    frameStart := 0 },
  { event := event284521
    frameStart := 0 },
  { event := event284522
    frameStart := 0 },
  { event := event284523
    frameStart := 0 },
  { event := event284524
    frameStart := 0 },
  { event := event284525
    frameStart := 0 },
  { event := event284526
    frameStart := 0 },
  { event := event284527
    frameStart := 0 }
]

def eventLeaf17783 : Array AnnotatedEvent := #[
  { event := event284528
    frameStart := 0 },
  { event := event284529
    frameStart := 0 },
  { event := event284530
    frameStart := 0 },
  { event := event284531
    frameStart := 0 },
  { event := event284532
    frameStart := 0 },
  { event := event284533
    frameStart := 0 },
  { event := event284534
    frameStart := 0 },
  { event := event284535
    frameStart := 0 },
  { event := event284536
    frameStart := 0 },
  { event := event284537
    frameStart := 0 },
  { event := event284538
    frameStart := 0 },
  { event := event284539
    frameStart := 0 },
  { event := event284540
    frameStart := 0 },
  { event := event284541
    frameStart := 0 },
  { event := event284542
    frameStart := 0 },
  { event := event284543
    frameStart := 0 }
]

def eventLeaf17784 : Array AnnotatedEvent := #[
  { event := event284544
    frameStart := 0 },
  { event := event284545
    frameStart := 0 },
  { event := event284546
    frameStart := 0 },
  { event := event284547
    frameStart := 0 },
  { event := event284548
    frameStart := 0 },
  { event := event284549
    frameStart := 0 },
  { event := event284550
    frameStart := 0 },
  { event := event284551
    frameStart := 0 },
  { event := event284552
    frameStart := 0 },
  { event := event284553
    frameStart := 0 },
  { event := event284554
    frameStart := 0 },
  { event := event284555
    frameStart := 0 },
  { event := event284556
    frameStart := 0 },
  { event := event284557
    frameStart := 0 },
  { event := event284558
    frameStart := 0 },
  { event := event284559
    frameStart := 0 }
]

def eventLeaf17785 : Array AnnotatedEvent := #[
  { event := event284560
    frameStart := 0 },
  { event := event284561
    frameStart := 0 },
  { event := event284562
    frameStart := 0 },
  { event := event284563
    frameStart := 0 },
  { event := event284564
    frameStart := 0 },
  { event := event284565
    frameStart := 0 },
  { event := event284566
    frameStart := 0 },
  { event := event284567
    frameStart := 0 },
  { event := event284568
    frameStart := 0 },
  { event := event284569
    frameStart := 0 },
  { event := event284570
    frameStart := 0 },
  { event := event284571
    frameStart := 0 },
  { event := event284572
    frameStart := 0 },
  { event := event284573
    frameStart := 0 },
  { event := event284574
    frameStart := 0 },
  { event := event284575
    frameStart := 0 }
]

def eventLeaf17786 : Array AnnotatedEvent := #[
  { event := event284576
    frameStart := 0 },
  { event := event284577
    frameStart := 0 },
  { event := event284578
    frameStart := 0 },
  { event := event284579
    frameStart := 0 },
  { event := event284580
    frameStart := 0 },
  { event := event284581
    frameStart := 0 },
  { event := event284582
    frameStart := 0 },
  { event := event284583
    frameStart := 0 },
  { event := event284584
    frameStart := 0 },
  { event := event284585
    frameStart := 0 },
  { event := event284586
    frameStart := 0 },
  { event := event284587
    frameStart := 0 },
  { event := event284588
    frameStart := 0 },
  { event := event284589
    frameStart := 0 },
  { event := event284590
    frameStart := 0 },
  { event := event284591
    frameStart := 0 }
]

def eventLeaf17787 : Array AnnotatedEvent := #[
  { event := event284592
    frameStart := 284592 },
  { event := event284593
    frameStart := 284592 },
  { event := event284594
    frameStart := 284592 },
  { event := event284595
    frameStart := 284592 },
  { event := event284596
    frameStart := 284592 },
  { event := event284597
    frameStart := 284592 },
  { event := event284598
    frameStart := 284592 },
  { event := event284599
    frameStart := 284592 },
  { event := event284600
    frameStart := 284592 },
  { event := event284601
    frameStart := 284592 },
  { event := event284602
    frameStart := 284592 },
  { event := event284603
    frameStart := 284592 },
  { event := event284604
    frameStart := 284592 },
  { event := event284605
    frameStart := 284592 },
  { event := event284606
    frameStart := 284592 },
  { event := event284607
    frameStart := 284592 }
]

def eventLeaf17788 : Array AnnotatedEvent := #[
  { event := event284608
    frameStart := 284592 },
  { event := event284609
    frameStart := 284592 },
  { event := event284610
    frameStart := 284592 },
  { event := event284611
    frameStart := 284592 },
  { event := event284612
    frameStart := 284592 },
  { event := event284613
    frameStart := 284592 },
  { event := event284614
    frameStart := 284592 },
  { event := event284615
    frameStart := 284592 },
  { event := event284616
    frameStart := 284592 },
  { event := event284617
    frameStart := 284592 },
  { event := event284618
    frameStart := 284592 },
  { event := event284619
    frameStart := 284592 },
  { event := event284620
    frameStart := 284592 },
  { event := event284621
    frameStart := 284592 },
  { event := event284622
    frameStart := 284592 },
  { event := event284623
    frameStart := 284592 }
]

def eventLeaf17789 : Array AnnotatedEvent := #[
  { event := event284624
    frameStart := 284592 },
  { event := event284625
    frameStart := 284592 },
  { event := event284626
    frameStart := 284592 },
  { event := event284627
    frameStart := 284592 },
  { event := event284628
    frameStart := 284592 },
  { event := event284629
    frameStart := 284592 },
  { event := event284630
    frameStart := 284592 },
  { event := event284631
    frameStart := 284592 },
  { event := event284632
    frameStart := 284592 },
  { event := event284633
    frameStart := 284592 },
  { event := event284634
    frameStart := 284592 },
  { event := event284635
    frameStart := 284592 },
  { event := event284636
    frameStart := 284592 },
  { event := event284637
    frameStart := 284592 },
  { event := event284638
    frameStart := 284592 },
  { event := event284639
    frameStart := 284592 }
]

def eventLeaf17790 : Array AnnotatedEvent := #[
  { event := event284640
    frameStart := 284640 },
  { event := event284641
    frameStart := 284640 },
  { event := event284642
    frameStart := 284640 },
  { event := event284643
    frameStart := 284640 },
  { event := event284644
    frameStart := 284640 },
  { event := event284645
    frameStart := 284640 },
  { event := event284646
    frameStart := 284640 },
  { event := event284647
    frameStart := 284640 },
  { event := event284648
    frameStart := 284640 },
  { event := event284649
    frameStart := 284640 },
  { event := event284650
    frameStart := 284640 },
  { event := event284651
    frameStart := 284640 },
  { event := event284652
    frameStart := 284640 },
  { event := event284653
    frameStart := 284640 },
  { event := event284654
    frameStart := 284640 },
  { event := event284655
    frameStart := 284640 }
]

def eventLeaf17791 : Array AnnotatedEvent := #[
  { event := event284656
    frameStart := 284640 },
  { event := event284657
    frameStart := 284640 },
  { event := event284658
    frameStart := 284640 },
  { event := event284659
    frameStart := 284640 },
  { event := event284660
    frameStart := 284640 },
  { event := event284661
    frameStart := 284640 },
  { event := event284662
    frameStart := 284640 },
  { event := event284663
    frameStart := 284640 },
  { event := event284664
    frameStart := 284640 },
  { event := event284665
    frameStart := 284640 },
  { event := event284666
    frameStart := 284640 },
  { event := event284667
    frameStart := 284640 },
  { event := event284668
    frameStart := 284640 },
  { event := event284669
    frameStart := 284640 },
  { event := event284670
    frameStart := 284640 },
  { event := event284671
    frameStart := 284640 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1111
