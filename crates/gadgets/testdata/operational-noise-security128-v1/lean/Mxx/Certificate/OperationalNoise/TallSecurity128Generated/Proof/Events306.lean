import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events306

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34582⟩⟩) 1 ⟨34581⟩ 78329

def event78337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34582⟩⟩) (.sum [.predecessor 0 78335 .coefficient, .predecessor 1 78336 .coefficient])

def exact78338RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78338RawTermsValid :
    exact78338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34582⟩⟩) exact78338RawTerms .large 78337 .exactZero (none)

def event78339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34583⟩⟩) 0 ⟨34582⟩ 78338

def event78340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34583⟩⟩) 1 ⟨106⟩ 19577

def event78341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34583⟩⟩) (.sum [.predecessor 0 78339 .coefficient, .predecessor 1 78340 .coefficient])

def event78342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34583⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨106⟩⟩]⟩) [⟨.result 19577 .coefficient, false, none⟩])

def event78343 : Event := .survivorFold (1) 78342

def exact78344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78344RawTermsValid :
    exact78344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34583⟩⟩) exact78344RawTerms .large 78341 (.finite 26) (some (78342))

def event78345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34584⟩⟩) 0 ⟨34583⟩ 78344

def event78346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34584⟩⟩) 1 ⟨13671⟩ 3204

def event78347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34584⟩⟩) (.product (.predecessor 0 78345 .coefficient) (.predecessor 1 78346 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34584⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩) [⟨.result 3204 .coefficient, true, some 1⟩])

def event78349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34584⟩⟩) (.product (.result 78344 .summary) (.transfer 78348) (⟨false, false, none, none, none⟩))

def event78350 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34584⟩⟩, .operator (⟨78344, 1⟩, ⟨3204, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event78351 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34584⟩⟩, .operator (⟨78344, 0⟩, ⟨3204, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def exact78352RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78352RawTermsValid :
    exact78352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34584⟩⟩) exact78352RawTerms .large 78347 (.finite 34078720) (some (78349))

def event78353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13672⟩⟩) 0 ⟨13671⟩ 3204

def event78354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13672⟩⟩) 1 ⟨10328⟩ 75903

def event78355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13672⟩⟩) (.tensor (.predecessor 0 78353 .coefficient) (.predecessor 1 78354 .coefficient) true false)

def event78356 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13672⟩⟩, .operator (⟨3204, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78357RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78357RawTermsValid :
    exact78357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13672⟩⟩) exact78357RawTerms .large 78355 .exactZero (none)

def event78358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10355⟩⟩) 0 ⟨10327⟩ 75773

def event78359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10355⟩⟩) 1 ⟨7297⟩ 19626

def event78360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10355⟩⟩) (.product (.predecessor 0 78358 .coefficient) (.predecessor 1 78359 .coefficient) (⟨false, false, none, none, none⟩))

def event78361 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10355⟩⟩, .operator (⟨75773, 0⟩, ⟨19626, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩)

def exact78362RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩]

theorem exact78362RawTermsValid :
    exact78362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10355⟩⟩) exact78362RawTerms .large 78360 .exactZero (none)

def event78363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13673⟩⟩) 0 ⟨10355⟩ 78362

def event78364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13673⟩⟩) 1 ⟨13672⟩ 78357

def event78365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13673⟩⟩) (.sum [.predecessor 0 78363 .coefficient, .predecessor 1 78364 .coefficient])

def exact78366RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78366RawTermsValid :
    exact78366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13673⟩⟩) exact78366RawTerms .large 78365 .exactZero (none)

def event78367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13674⟩⟩) 0 ⟨13673⟩ 78366

def event78368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13674⟩⟩) 1 ⟨123⟩ 19618

def event78369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13674⟩⟩) (.sum [.predecessor 0 78367 .coefficient, .predecessor 1 78368 .coefficient])

def event78370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13674⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨123⟩⟩]⟩) [⟨.result 19618 .coefficient, false, none⟩])

def event78371 : Event := .survivorFold (1) 78370

def exact78372RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78372RawTermsValid :
    exact78372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78372 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13674⟩⟩) exact78372RawTerms .large 78369 (.finite 26) (some (78370))

def event78373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13675⟩⟩) 0 ⟨13674⟩ 78372

def event78374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13675⟩⟩) 1 ⟨9551⟩ 19615

def event78375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13675⟩⟩) (.product (.predecessor 0 78373 .coefficient) (.predecessor 1 78374 .coefficient) (⟨false, false, none, none, none⟩))

def event78376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13675⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) [⟨.result 19611 .coefficient, false, none⟩])

def event78377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13675⟩⟩) (.product (.result 78372 .summary) (.transfer 78376) (⟨false, false, none, none, none⟩))

def event78378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13675⟩⟩, .operator (⟨78372, 1⟩, ⟨19615, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (-1)⟩)

def event78379 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13675⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9550⟩⟩) ⟨7280⟩ 19585)

def event78380 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13675⟩⟩, .relation 78379 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (-1)⟩)

def event78381 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13675⟩⟩, .operator (⟨78372, 0⟩, ⟨19615, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩)

def exact78382RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (-1)⟩]

theorem exact78382RawTermsValid :
    exact78382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13675⟩⟩) exact78382RawTerms .large 78375 (.finite 279172874240) (some (78377))

def event78383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34585⟩⟩) 0 ⟨13675⟩ 78382

def event78384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34585⟩⟩) 1 ⟨34584⟩ 78352

def event78385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34585⟩⟩) (.sum [.predecessor 0 78383 .coefficient, .predecessor 1 78384 .coefficient])

def event78386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34585⟩⟩, .operator (⟨78382, 1⟩, ⟨78352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def event78387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34585⟩⟩) (.sum [.result 78382 .summary, .result 78352 .summary])

def exact78388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78388RawTermsValid :
    exact78388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34585⟩⟩) exact78388RawTerms .large 78385 (.finite 279206952960) (some (78387))

def event78389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36326⟩⟩) 0 ⟨34585⟩ 78388

def event78390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36326⟩⟩) 1 ⟨36325⟩ 78324

def event78391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36326⟩⟩) (.product (.predecessor 0 78389 .coefficient) (.predecessor 1 78390 .coefficient) (⟨false, false, none, none, none⟩))

def event78392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36326⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩) [⟨.result 78324 .coefficient, false, none⟩])

def event78393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36326⟩⟩) (.product (.result 78388 .summary) (.transfer 78392) (⟨false, false, none, none, none⟩))

def event78394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36326⟩⟩, .operator (⟨78388, 1⟩, ⟨78324, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩)

def event78395 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36326⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36325⟩⟩) ⟨35785⟩ 78321)

def event78396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36326⟩⟩, .relation 78395 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (-1)⟩)

def event78397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36326⟩⟩, .operator (⟨78388, 0⟩, ⟨78324, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩)

def exact78398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (-1)⟩]

theorem exact78398RawTermsValid :
    exact78398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36326⟩⟩) exact78398RawTerms .large 78391 (.finite 2997961829447525990400) (some (78393))

def event78399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35249⟩⟩) 0 ⟨34580⟩ 3212

def event78400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35249⟩⟩) (.authority (.relationPreimageSource ⟨49⟩))

def exact78401RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩]

theorem exact78401RawTermsValid :
    exact78401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35249⟩⟩) exact78401RawTerms (.finite 5647228698) 78400 .exactZero (none)

def event78402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35251⟩⟩) 0 ⟨35249⟩ 78401

def event78403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35251⟩⟩) 1 ⟨2370⟩ 4

def event78404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35251⟩⟩) (.scale (.predecessor 0 78402 .coefficient) (.value (.predecessor 1 78403 .coefficient)))

def exact78405RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩]

theorem exact78405RawTermsValid :
    exact78405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35251⟩⟩) exact78405RawTerms (.finite 5647228698) 78404 .exactZero (none)

def event78406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35252⟩⟩) 0 ⟨10368⟩ 75995

def event78407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35252⟩⟩) 1 ⟨35251⟩ 78405

def event78408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35252⟩⟩) (.product (.predecessor 0 78406 .coefficient) (.predecessor 1 78407 .coefficient) (⟨false, false, none, none, none⟩))

def event78409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35252⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩) [⟨.result 78401 .coefficient, false, none⟩])

def event78410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35252⟩⟩) (.product (.result 75995 .summary) (.transfer 78409) (⟨false, false, none, none, none⟩))

def event78411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35252⟩⟩, .operator (⟨75995, 0⟩, ⟨78405, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩)

def event78412 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35250⟩⟩)

def event78413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78414 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78416 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78418 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78420 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78420

def event78422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78418

def event78423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78421 .coefficient) (.value (.predecessor 1 78422 .coefficient)))

def event78424 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78424

def event78426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78416

def event78427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78425 .coefficient, .predecessor 1 78426 .coefficient])

def event78428 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78428

def event78430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78414

def event78431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78430 .coefficient))

def event78432 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 78432

def event78434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact78435RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78435RawTermsValid :
    exact78435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact78435RawTerms (.finite 40) 78434 .exactZero (none)

def event78436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 78432

def event78437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact78438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact78438RawTermsValid :
    exact78438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact78438RawTerms (.finite 40) 78437 .exactZero (none)

def event78439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 78438

def event78440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 78435

def event78441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 78439 .coefficient) (.predecessor 1 78440 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩) [⟨.result 78438 .coefficient, true, some 1⟩, ⟨.result 78435 .coefficient, true, some 1⟩])

def event78443 : Event := .survivorFold (1) 78442

def exact78444RawTerms : List Term := []

theorem exact78444RawTermsValid :
    exact78444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact78444RawTerms (.finite 1600) 78441 (.finite 1600) (some (78442))

def event78445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 78444

def event78446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 78445 .coefficient))

def event78447 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event78448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35249⟩⟩) 0 ⟨34580⟩ 78447

def event78449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35249⟩⟩) (.authority (.relationPreimageSource ⟨49⟩))

def exact78450RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩]

theorem exact78450RawTermsValid :
    exact78450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35249⟩⟩) exact78450RawTerms (.finite 5647228698) 78449 .exactZero (none)

def event78451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact78452RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact78452RawTermsValid :
    exact78452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact78452RawTerms .large 78451 .exactZero (none)

def event78453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35250⟩⟩) 0 ⟨35⟩ 78452

def event78454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35250⟩⟩) 1 ⟨35249⟩ 78450

def event78455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35250⟩⟩) (.product (.predecessor 0 78453 .coefficient) (.predecessor 1 78454 .coefficient) (⟨false, false, none, none, none⟩))

def event78456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35250⟩⟩, .operator (⟨78452, 0⟩, ⟨78450, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩)

def exact78457RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩]

theorem exact78457RawTermsValid :
    exact78457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35250⟩⟩) exact78457RawTerms .large 78455 .exactZero (none)

def event78458 : Event := .preFoldPolynomial 78457 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩] .exactZero none

def exact78459RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩, (1)⟩]

def event78459 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35250⟩⟩) 78458 exact78459RawTerms .large 78455 .exactZero (none)

def event78460 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36329⟩⟩)

def event78461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78462 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78464 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78466 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78468 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78468

def event78470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78466

def event78471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78469 .coefficient) (.value (.predecessor 1 78470 .coefficient)))

def event78472 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78472

def event78474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78464

def event78475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78473 .coefficient, .predecessor 1 78474 .coefficient])

def event78476 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78476

def event78478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78462

def event78479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78478 .coefficient))

def event78480 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 78480

def event78482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact78483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78483RawTermsValid :
    exact78483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact78483RawTerms (.finite 40) 78482 .exactZero (none)

def event78484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 78480

def event78485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact78486RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact78486RawTermsValid :
    exact78486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact78486RawTerms (.finite 40) 78485 .exactZero (none)

def event78487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 78486

def event78488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 78483

def event78489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 78487 .coefficient) (.predecessor 1 78488 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78490 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34579⟩⟩, .operator (⟨78486, 0⟩, ⟨78483, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩)

def exact78491RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78491RawTermsValid :
    exact78491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78491 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact78491RawTerms (.finite 1600) 78489 .exactZero (none)

def event78492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 78491

def event78493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 78492 .coefficient))

def event78494 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event78495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35784⟩⟩) 0 ⟨34580⟩ 78494

def event78496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35784⟩⟩) (.authority (.programFamilyFact))

def event78497 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35784⟩⟩) (.finite 3720)

def event78498 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event78499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35785⟩⟩) 0 ⟨7177⟩ 78498

def event78500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35785⟩⟩) 1 ⟨35784⟩ 78497

def event78501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35785⟩⟩) (.authority (.operator))

def exact78502RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩]

theorem exact78502RawTermsValid :
    exact78502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78502 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35785⟩⟩) exact78502RawTerms .large 78501 .exactZero (none)

def event78503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36325⟩⟩) 0 ⟨35785⟩ 78502

def event78504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36325⟩⟩) (.authority (.operator))

def exact78505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩]

theorem exact78505RawTermsValid :
    exact78505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78505 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36325⟩⟩) exact78505RawTerms (.finite 8192) 78504 .exactZero (none)

def event78506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event78507 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event78508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36050⟩⟩) 0 ⟨34580⟩ 78494

def event78509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36050⟩⟩) 1 ⟨136⟩ 78507

def event78510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36050⟩⟩) (.sum [.predecessor 0 78508 .coefficient, .predecessor 1 78509 .coefficient])

def event78511 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36050⟩⟩) (.finite 1600)

def event78512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36051⟩⟩) 0 ⟨36050⟩ 78511

def event78513 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36051⟩⟩) (.identity (.predecessor 0 78512 .coefficient))

def exact78514RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78514RawTermsValid :
    exact78514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36051⟩⟩) exact78514RawTerms (.finite 1600) 78513 .exactZero (none)

def event78515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact78516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78516RawTermsValid :
    exact78516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact78516RawTerms .large 78515 .exactZero (none)

def event78517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36052⟩⟩) 0 ⟨6908⟩ 78516

def event78518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36052⟩⟩) 1 ⟨36051⟩ 78514

def event78519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36052⟩⟩) (.product (.predecessor 0 78517 .coefficient) (.predecessor 1 78518 .coefficient) (⟨false, false, none, none, none⟩))

def event78520 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36052⟩⟩, .operator (⟨78516, 0⟩, ⟨78514, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78521RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78521RawTermsValid :
    exact78521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78521 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36052⟩⟩) exact78521RawTerms .large 78519 .exactZero (none)

def event78522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event78523 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event78524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 78498

def event78525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact78526RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact78526RawTermsValid :
    exact78526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact78526RawTerms .large 78525 .exactZero (none)

def event78527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7280⟩⟩) 0 ⟨7178⟩ 78526

def event78528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7280⟩⟩) (.identity (.predecessor 0 78527 .coefficient))

def exact78529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩]

theorem exact78529RawTermsValid :
    exact78529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7280⟩⟩) exact78529RawTerms .large 78528 .exactZero (none)

def event78530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9550⟩⟩) 0 ⟨7280⟩ 78529

def event78531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9550⟩⟩) (.authority (.operator))

def exact78532RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact78532RawTermsValid :
    exact78532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9550⟩⟩) exact78532RawTerms (.finite 8192) 78531 .exactZero (none)

def event78533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9551⟩⟩) 0 ⟨9550⟩ 78532

def event78534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9551⟩⟩) 1 ⟨2370⟩ 78523

def event78535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9551⟩⟩) (.scale (.predecessor 0 78533 .coefficient) (.value (.predecessor 1 78534 .coefficient)))

def exact78536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact78536RawTermsValid :
    exact78536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78536 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9551⟩⟩) exact78536RawTerms (.finite 8192) 78535 .exactZero (none)

def event78537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7297⟩⟩) 0 ⟨7178⟩ 78526

def event78538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7297⟩⟩) (.identity (.predecessor 0 78537 .coefficient))

def exact78539RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩]

theorem exact78539RawTermsValid :
    exact78539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78539 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7297⟩⟩) exact78539RawTerms .large 78538 .exactZero (none)

def event78540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9552⟩⟩) 0 ⟨7297⟩ 78539

def event78541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9552⟩⟩) 1 ⟨9551⟩ 78536

def event78542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9552⟩⟩) (.product (.predecessor 0 78540 .coefficient) (.predecessor 1 78541 .coefficient) (⟨false, false, none, none, none⟩))

def event78543 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9552⟩⟩, .operator (⟨78539, 0⟩, ⟨78536, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩)

def exact78544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact78544RawTermsValid :
    exact78544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9552⟩⟩) exact78544RawTerms .large 78542 .exactZero (none)

def event78545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36053⟩⟩) 0 ⟨9552⟩ 78544

def event78546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36053⟩⟩) 1 ⟨36052⟩ 78521

def event78547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36053⟩⟩) (.sum [.predecessor 0 78545 .coefficient, .predecessor 1 78546 .coefficient])

def exact78548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78548RawTermsValid :
    exact78548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36053⟩⟩) exact78548RawTerms .large 78547 .exactZero (none)

def event78549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36328⟩⟩) 0 ⟨36053⟩ 78548

def event78550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36328⟩⟩) 1 ⟨36325⟩ 78505

def event78551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36328⟩⟩) (.product (.predecessor 0 78549 .coefficient) (.predecessor 1 78550 .coefficient) (⟨false, false, none, none, none⟩))

def event78552 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36328⟩⟩, .operator (⟨78548, 0⟩, ⟨78505, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩)

def event78553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36328⟩⟩, .operator (⟨78548, 1⟩, ⟨78505, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩)

def event78554 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36328⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36325⟩⟩) ⟨35785⟩ 78502)

def event78555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36328⟩⟩, .relation 78554 0, ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (-1)⟩)

def exact78556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (-1)⟩]

theorem exact78556RawTermsValid :
    exact78556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36328⟩⟩) exact78556RawTerms .large 78551 .exactZero (none)

def event78557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34796⟩⟩) 0 ⟨34580⟩ 78494

def event78558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34796⟩⟩) (.authority (.programFamilyFact))

def exact78559RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact78559RawTermsValid :
    exact78559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78559 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34796⟩⟩) exact78559RawTerms (.finite 40) 78558 .exactZero (none)

def event78560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34798⟩⟩) 0 ⟨6908⟩ 78516

def event78561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34798⟩⟩) 1 ⟨34796⟩ 78559

def event78562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34798⟩⟩) (.product (.predecessor 0 78560 .coefficient) (.predecessor 1 78561 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34798⟩⟩, .operator (⟨78516, 0⟩, ⟨78559, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78564RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78564RawTermsValid :
    exact78564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78564 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34798⟩⟩) exact78564RawTerms .large 78562 .exactZero (none)

def event78565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 78498

def event78566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact78567RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact78567RawTermsValid :
    exact78567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78567 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact78567RawTerms .large 78566 .exactZero (none)

def event78568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34799⟩⟩) 0 ⟨7191⟩ 78567

def event78569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34799⟩⟩) 1 ⟨34798⟩ 78564

def event78570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34799⟩⟩) (.sum [.predecessor 0 78568 .coefficient, .predecessor 1 78569 .coefficient])

def exact78571RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78571RawTermsValid :
    exact78571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34799⟩⟩) exact78571RawTerms .large 78570 .exactZero (none)

def event78572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36329⟩⟩) 0 ⟨34799⟩ 78571

def event78573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36329⟩⟩) 1 ⟨36328⟩ 78556

def event78574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36329⟩⟩) (.sum [.predecessor 0 78572 .coefficient, .predecessor 1 78573 .coefficient])

def exact78575RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78575RawTermsValid :
    exact78575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36329⟩⟩) exact78575RawTerms .large 78574 .exactZero (none)

def event78576 : Event := .preFoldPolynomial 78575 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event78577 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36329⟩⟩) 78576 exact78577RawTerms .large 78574 .exactZero (none)

def event78578 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34580⟩⟩) ⟨⟨70⟩, ⟨49⟩, ⟨135⟩⟩ ⟨78412, 78578⟩

def event78579 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35252⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩) (1) 0 2 (.universal 78578 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35249⟩⟩]⟩) (none) 78577)

def event78580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35252⟩⟩, .relation 78579 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩)

def event78581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35252⟩⟩, .relation 78579 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩)

def event78582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35252⟩⟩, .relation 78579 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩)

def event78583 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35252⟩⟩, .relation 78579 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact78584RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78584RawTermsValid :
    exact78584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35252⟩⟩) exact78584RawTerms .large 78408 (.finite 202072841853861888) (some (78410))

def event78585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36327⟩⟩) 0 ⟨35252⟩ 78584

def event78586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36327⟩⟩) 1 ⟨36326⟩ 78398

def event78587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36327⟩⟩) (.sum [.predecessor 0 78585 .coefficient, .predecessor 1 78586 .coefficient])

def event78588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36327⟩⟩, .operator (⟨78584, 2⟩, ⟨78398, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (-1)⟩)

def event78589 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36327⟩⟩, .operator (⟨78584, 1⟩, ⟨78398, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩)

def event78590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36327⟩⟩) (.sum [.result 78584 .summary, .result 78398 .summary])

def exact78591RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78591RawTermsValid :
    exact78591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78591 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36327⟩⟩) exact78591RawTerms .large 78587 (.finite 2998163902289379852288) (some (78590))

def eventLeaf4896 : Array AnnotatedEvent := #[
  { event := event78336
    frameStart := 0 },
  { event := event78337
    frameStart := 0 },
  { event := event78338
    frameStart := 0 },
  { event := event78339
    frameStart := 0 },
  { event := event78340
    frameStart := 0 },
  { event := event78341
    frameStart := 0 },
  { event := event78342
    frameStart := 0 },
  { event := event78343
    frameStart := 0 },
  { event := event78344
    frameStart := 0 },
  { event := event78345
    frameStart := 0 },
  { event := event78346
    frameStart := 0 },
  { event := event78347
    frameStart := 0 },
  { event := event78348
    frameStart := 0 },
  { event := event78349
    frameStart := 0 },
  { event := event78350
    frameStart := 0 },
  { event := event78351
    frameStart := 0 }
]

def eventLeaf4897 : Array AnnotatedEvent := #[
  { event := event78352
    frameStart := 0 },
  { event := event78353
    frameStart := 0 },
  { event := event78354
    frameStart := 0 },
  { event := event78355
    frameStart := 0 },
  { event := event78356
    frameStart := 0 },
  { event := event78357
    frameStart := 0 },
  { event := event78358
    frameStart := 0 },
  { event := event78359
    frameStart := 0 },
  { event := event78360
    frameStart := 0 },
  { event := event78361
    frameStart := 0 },
  { event := event78362
    frameStart := 0 },
  { event := event78363
    frameStart := 0 },
  { event := event78364
    frameStart := 0 },
  { event := event78365
    frameStart := 0 },
  { event := event78366
    frameStart := 0 },
  { event := event78367
    frameStart := 0 }
]

def eventLeaf4898 : Array AnnotatedEvent := #[
  { event := event78368
    frameStart := 0 },
  { event := event78369
    frameStart := 0 },
  { event := event78370
    frameStart := 0 },
  { event := event78371
    frameStart := 0 },
  { event := event78372
    frameStart := 0 },
  { event := event78373
    frameStart := 0 },
  { event := event78374
    frameStart := 0 },
  { event := event78375
    frameStart := 0 },
  { event := event78376
    frameStart := 0 },
  { event := event78377
    frameStart := 0 },
  { event := event78378
    frameStart := 0 },
  { event := event78379
    frameStart := 0 },
  { event := event78380
    frameStart := 0 },
  { event := event78381
    frameStart := 0 },
  { event := event78382
    frameStart := 0 },
  { event := event78383
    frameStart := 0 }
]

def eventLeaf4899 : Array AnnotatedEvent := #[
  { event := event78384
    frameStart := 0 },
  { event := event78385
    frameStart := 0 },
  { event := event78386
    frameStart := 0 },
  { event := event78387
    frameStart := 0 },
  { event := event78388
    frameStart := 0 },
  { event := event78389
    frameStart := 0 },
  { event := event78390
    frameStart := 0 },
  { event := event78391
    frameStart := 0 },
  { event := event78392
    frameStart := 0 },
  { event := event78393
    frameStart := 0 },
  { event := event78394
    frameStart := 0 },
  { event := event78395
    frameStart := 0 },
  { event := event78396
    frameStart := 0 },
  { event := event78397
    frameStart := 0 },
  { event := event78398
    frameStart := 0 },
  { event := event78399
    frameStart := 0 }
]

def eventLeaf4900 : Array AnnotatedEvent := #[
  { event := event78400
    frameStart := 0 },
  { event := event78401
    frameStart := 0 },
  { event := event78402
    frameStart := 0 },
  { event := event78403
    frameStart := 0 },
  { event := event78404
    frameStart := 0 },
  { event := event78405
    frameStart := 0 },
  { event := event78406
    frameStart := 0 },
  { event := event78407
    frameStart := 0 },
  { event := event78408
    frameStart := 0 },
  { event := event78409
    frameStart := 0 },
  { event := event78410
    frameStart := 0 },
  { event := event78411
    frameStart := 0 },
  { event := event78412
    frameStart := 78412 },
  { event := event78413
    frameStart := 78412 },
  { event := event78414
    frameStart := 78412 },
  { event := event78415
    frameStart := 78412 }
]

def eventLeaf4901 : Array AnnotatedEvent := #[
  { event := event78416
    frameStart := 78412 },
  { event := event78417
    frameStart := 78412 },
  { event := event78418
    frameStart := 78412 },
  { event := event78419
    frameStart := 78412 },
  { event := event78420
    frameStart := 78412 },
  { event := event78421
    frameStart := 78412 },
  { event := event78422
    frameStart := 78412 },
  { event := event78423
    frameStart := 78412 },
  { event := event78424
    frameStart := 78412 },
  { event := event78425
    frameStart := 78412 },
  { event := event78426
    frameStart := 78412 },
  { event := event78427
    frameStart := 78412 },
  { event := event78428
    frameStart := 78412 },
  { event := event78429
    frameStart := 78412 },
  { event := event78430
    frameStart := 78412 },
  { event := event78431
    frameStart := 78412 }
]

def eventLeaf4902 : Array AnnotatedEvent := #[
  { event := event78432
    frameStart := 78412 },
  { event := event78433
    frameStart := 78412 },
  { event := event78434
    frameStart := 78412 },
  { event := event78435
    frameStart := 78412 },
  { event := event78436
    frameStart := 78412 },
  { event := event78437
    frameStart := 78412 },
  { event := event78438
    frameStart := 78412 },
  { event := event78439
    frameStart := 78412 },
  { event := event78440
    frameStart := 78412 },
  { event := event78441
    frameStart := 78412 },
  { event := event78442
    frameStart := 78412 },
  { event := event78443
    frameStart := 78412 },
  { event := event78444
    frameStart := 78412 },
  { event := event78445
    frameStart := 78412 },
  { event := event78446
    frameStart := 78412 },
  { event := event78447
    frameStart := 78412 }
]

def eventLeaf4903 : Array AnnotatedEvent := #[
  { event := event78448
    frameStart := 78412 },
  { event := event78449
    frameStart := 78412 },
  { event := event78450
    frameStart := 78412 },
  { event := event78451
    frameStart := 78412 },
  { event := event78452
    frameStart := 78412 },
  { event := event78453
    frameStart := 78412 },
  { event := event78454
    frameStart := 78412 },
  { event := event78455
    frameStart := 78412 },
  { event := event78456
    frameStart := 78412 },
  { event := event78457
    frameStart := 78412 },
  { event := event78458
    frameStart := 78412 },
  { event := event78459
    frameStart := 78412 },
  { event := event78460
    frameStart := 78460 },
  { event := event78461
    frameStart := 78460 },
  { event := event78462
    frameStart := 78460 },
  { event := event78463
    frameStart := 78460 }
]

def eventLeaf4904 : Array AnnotatedEvent := #[
  { event := event78464
    frameStart := 78460 },
  { event := event78465
    frameStart := 78460 },
  { event := event78466
    frameStart := 78460 },
  { event := event78467
    frameStart := 78460 },
  { event := event78468
    frameStart := 78460 },
  { event := event78469
    frameStart := 78460 },
  { event := event78470
    frameStart := 78460 },
  { event := event78471
    frameStart := 78460 },
  { event := event78472
    frameStart := 78460 },
  { event := event78473
    frameStart := 78460 },
  { event := event78474
    frameStart := 78460 },
  { event := event78475
    frameStart := 78460 },
  { event := event78476
    frameStart := 78460 },
  { event := event78477
    frameStart := 78460 },
  { event := event78478
    frameStart := 78460 },
  { event := event78479
    frameStart := 78460 }
]

def eventLeaf4905 : Array AnnotatedEvent := #[
  { event := event78480
    frameStart := 78460 },
  { event := event78481
    frameStart := 78460 },
  { event := event78482
    frameStart := 78460 },
  { event := event78483
    frameStart := 78460 },
  { event := event78484
    frameStart := 78460 },
  { event := event78485
    frameStart := 78460 },
  { event := event78486
    frameStart := 78460 },
  { event := event78487
    frameStart := 78460 },
  { event := event78488
    frameStart := 78460 },
  { event := event78489
    frameStart := 78460 },
  { event := event78490
    frameStart := 78460 },
  { event := event78491
    frameStart := 78460 },
  { event := event78492
    frameStart := 78460 },
  { event := event78493
    frameStart := 78460 },
  { event := event78494
    frameStart := 78460 },
  { event := event78495
    frameStart := 78460 }
]

def eventLeaf4906 : Array AnnotatedEvent := #[
  { event := event78496
    frameStart := 78460 },
  { event := event78497
    frameStart := 78460 },
  { event := event78498
    frameStart := 78460 },
  { event := event78499
    frameStart := 78460 },
  { event := event78500
    frameStart := 78460 },
  { event := event78501
    frameStart := 78460 },
  { event := event78502
    frameStart := 78460 },
  { event := event78503
    frameStart := 78460 },
  { event := event78504
    frameStart := 78460 },
  { event := event78505
    frameStart := 78460 },
  { event := event78506
    frameStart := 78460 },
  { event := event78507
    frameStart := 78460 },
  { event := event78508
    frameStart := 78460 },
  { event := event78509
    frameStart := 78460 },
  { event := event78510
    frameStart := 78460 },
  { event := event78511
    frameStart := 78460 }
]

def eventLeaf4907 : Array AnnotatedEvent := #[
  { event := event78512
    frameStart := 78460 },
  { event := event78513
    frameStart := 78460 },
  { event := event78514
    frameStart := 78460 },
  { event := event78515
    frameStart := 78460 },
  { event := event78516
    frameStart := 78460 },
  { event := event78517
    frameStart := 78460 },
  { event := event78518
    frameStart := 78460 },
  { event := event78519
    frameStart := 78460 },
  { event := event78520
    frameStart := 78460 },
  { event := event78521
    frameStart := 78460 },
  { event := event78522
    frameStart := 78460 },
  { event := event78523
    frameStart := 78460 },
  { event := event78524
    frameStart := 78460 },
  { event := event78525
    frameStart := 78460 },
  { event := event78526
    frameStart := 78460 },
  { event := event78527
    frameStart := 78460 }
]

def eventLeaf4908 : Array AnnotatedEvent := #[
  { event := event78528
    frameStart := 78460 },
  { event := event78529
    frameStart := 78460 },
  { event := event78530
    frameStart := 78460 },
  { event := event78531
    frameStart := 78460 },
  { event := event78532
    frameStart := 78460 },
  { event := event78533
    frameStart := 78460 },
  { event := event78534
    frameStart := 78460 },
  { event := event78535
    frameStart := 78460 },
  { event := event78536
    frameStart := 78460 },
  { event := event78537
    frameStart := 78460 },
  { event := event78538
    frameStart := 78460 },
  { event := event78539
    frameStart := 78460 },
  { event := event78540
    frameStart := 78460 },
  { event := event78541
    frameStart := 78460 },
  { event := event78542
    frameStart := 78460 },
  { event := event78543
    frameStart := 78460 }
]

def eventLeaf4909 : Array AnnotatedEvent := #[
  { event := event78544
    frameStart := 78460 },
  { event := event78545
    frameStart := 78460 },
  { event := event78546
    frameStart := 78460 },
  { event := event78547
    frameStart := 78460 },
  { event := event78548
    frameStart := 78460 },
  { event := event78549
    frameStart := 78460 },
  { event := event78550
    frameStart := 78460 },
  { event := event78551
    frameStart := 78460 },
  { event := event78552
    frameStart := 78460 },
  { event := event78553
    frameStart := 78460 },
  { event := event78554
    frameStart := 78460 },
  { event := event78555
    frameStart := 78460 },
  { event := event78556
    frameStart := 78460 },
  { event := event78557
    frameStart := 78460 },
  { event := event78558
    frameStart := 78460 },
  { event := event78559
    frameStart := 78460 }
]

def eventLeaf4910 : Array AnnotatedEvent := #[
  { event := event78560
    frameStart := 78460 },
  { event := event78561
    frameStart := 78460 },
  { event := event78562
    frameStart := 78460 },
  { event := event78563
    frameStart := 78460 },
  { event := event78564
    frameStart := 78460 },
  { event := event78565
    frameStart := 78460 },
  { event := event78566
    frameStart := 78460 },
  { event := event78567
    frameStart := 78460 },
  { event := event78568
    frameStart := 78460 },
  { event := event78569
    frameStart := 78460 },
  { event := event78570
    frameStart := 78460 },
  { event := event78571
    frameStart := 78460 },
  { event := event78572
    frameStart := 78460 },
  { event := event78573
    frameStart := 78460 },
  { event := event78574
    frameStart := 78460 },
  { event := event78575
    frameStart := 78460 }
]

def eventLeaf4911 : Array AnnotatedEvent := #[
  { event := event78576
    frameStart := 78460 },
  { event := event78577
    frameStart := 78460 },
  { event := event78578
    frameStart := 0 },
  { event := event78579
    frameStart := 0 },
  { event := event78580
    frameStart := 0 },
  { event := event78581
    frameStart := 0 },
  { event := event78582
    frameStart := 0 },
  { event := event78583
    frameStart := 0 },
  { event := event78584
    frameStart := 0 },
  { event := event78585
    frameStart := 0 },
  { event := event78586
    frameStart := 0 },
  { event := event78587
    frameStart := 0 },
  { event := event78588
    frameStart := 0 },
  { event := event78589
    frameStart := 0 },
  { event := event78590
    frameStart := 0 },
  { event := event78591
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events306
