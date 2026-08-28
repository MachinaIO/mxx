import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events306

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78336 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27415⟩⟩, .operator (⟨78331, 2⟩, ⟨78153, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24032⟩⟩]⟩, (-1)⟩)

def event78337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27415⟩⟩) (.sum [.result 78331 .summary, .result 78153 .summary])

def exact78338RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78338RawTermsValid :
    exact78338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78338 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27415⟩⟩) exact78338RawTerms .large 78334 (.finite 1292001236604524572672) (some (78337))

def event78339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27416⟩⟩) 0 ⟨27415⟩ 78338

def event78340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27416⟩⟩) 1 ⟨6648⟩ 5759

def event78341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27416⟩⟩) (.product (.predecessor 0 78339 .coefficient) (.predecessor 1 78340 .coefficient) (⟨false, false, none, none, none⟩))

def event78342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27416⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩) [⟨.result 5755 .coefficient, false, none⟩])

def event78343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27416⟩⟩) (.product (.result 78338 .summary) (.transfer 78342) (⟨false, false, none, none, none⟩))

def event78344 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27416⟩⟩, .operator (⟨78338, 0⟩, ⟨5759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩)

def event78345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27416⟩⟩, .operator (⟨78338, 1⟩, ⟨5759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (-1)⟩)

def event78346 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27416⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6647⟩⟩) ⟨6595⟩ 5752)

def event78347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27416⟩⟩, .relation 78346 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78348RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78348RawTermsValid :
    exact78348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27416⟩⟩) exact78348RawTerms .large 78341 (.finite 4741665210358390854099402752) (some (78343))

def event78349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23969⟩⟩) 0 ⟨6689⟩ 5477

def event78350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23969⟩⟩) 1 ⟨23968⟩ 71555

def event78351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23969⟩⟩) (.authority (.operator))

def exact78352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩]

theorem exact78352RawTermsValid :
    exact78352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78352 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23969⟩⟩) exact78352RawTerms .large 78351 .exactZero (none)

def event78353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27195⟩⟩) 0 ⟨23969⟩ 78352

def event78354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27195⟩⟩) (.authority (.operator))

def exact78355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩]

theorem exact78355RawTermsValid :
    exact78355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78355 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27195⟩⟩) exact78355RawTerms (.finite 8192) 78354 .exactZero (none)

def event78356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27197⟩⟩) 0 ⟨25832⟩ 71839

def event78357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27197⟩⟩) 1 ⟨27195⟩ 78355

def event78358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27197⟩⟩) (.product (.predecessor 0 78356 .coefficient) (.predecessor 1 78357 .coefficient) (⟨false, false, none, none, none⟩))

def event78359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27197⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩) [⟨.result 78355 .coefficient, false, none⟩])

def event78360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27197⟩⟩) (.product (.result 71839 .summary) (.transfer 78359) (⟨false, false, none, none, none⟩))

def event78361 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27197⟩⟩, .operator (⟨71839, 0⟩, ⟨78355, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩)

def event78362 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27197⟩⟩, .operator (⟨71839, 1⟩, ⟨78355, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩)

def event78363 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27197⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27195⟩⟩) ⟨23969⟩ 78352)

def event78364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27197⟩⟩, .relation 78363 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (-1)⟩)

def exact78365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (-1)⟩]

theorem exact78365RawTermsValid :
    exact78365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27197⟩⟩) exact78365RawTerms .large 78358 (.finite 1291978822348200476672) (some (78360))

def event78366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20892⟩⟩) 0 ⟨15580⟩ 3402

def event78367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20892⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact78368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩]

theorem exact78368RawTermsValid :
    exact78368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20892⟩⟩) exact78368RawTerms (.finite 136065468) 78367 .exactZero (none)

def event78369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20894⟩⟩) 0 ⟨20892⟩ 78368

def event78370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20894⟩⟩) 1 ⟨2348⟩ 4

def event78371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20894⟩⟩) (.scale (.predecessor 0 78369 .coefficient) (.value (.predecessor 1 78370 .coefficient)))

def exact78372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩]

theorem exact78372RawTermsValid :
    exact78372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20894⟩⟩) exact78372RawTerms (.finite 136065468) 78371 .exactZero (none)

def event78373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20895⟩⟩) 0 ⟨5535⟩ 65387

def event78374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20895⟩⟩) 1 ⟨20894⟩ 78372

def event78375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20895⟩⟩) (.product (.predecessor 0 78373 .coefficient) (.predecessor 1 78374 .coefficient) (⟨false, false, none, none, none⟩))

def event78376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20895⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩) [⟨.result 78368 .coefficient, false, none⟩])

def event78377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20895⟩⟩) (.product (.result 65387 .summary) (.transfer 78376) (⟨false, false, none, none, none⟩))

def event78378 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20895⟩⟩, .operator (⟨65387, 0⟩, ⟨78372, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩)

def event78379 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20893⟩⟩)

def event78380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78381 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78383 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78387 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78387

def event78389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78385

def event78390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78388 .coefficient) (.value (.predecessor 1 78389 .coefficient)))

def event78391 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78391

def event78393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78383

def event78394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78392 .coefficient, .predecessor 1 78393 .coefficient])

def event78395 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78395

def event78397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78381

def event78398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78397 .coefficient))

def event78399 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 78399

def event78401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact78402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact78402RawTermsValid :
    exact78402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact78402RawTerms (.finite 10) 78401 .exactZero (none)

def event78403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 78399

def event78404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact78405RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact78405RawTermsValid :
    exact78405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact78405RawTerms (.finite 10) 78404 .exactZero (none)

def event78406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 78405

def event78407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 78402

def event78408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 78406 .coefficient) (.predecessor 1 78407 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩) [⟨.result 78405 .coefficient, true, some 1⟩, ⟨.result 78402 .coefficient, true, some 1⟩])

def event78410 : Event := .survivorFold (1) 78409

def exact78411RawTerms : List Term := []

theorem exact78411RawTermsValid :
    exact78411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact78411RawTerms (.finite 100) 78408 (.finite 100) (some (78409))

def event78412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 78411

def event78413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 78412 .coefficient))

def event78414 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event78415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15579⟩⟩) 0 ⟨13549⟩ 78414

def event78416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15579⟩⟩) (.authority (.programFamilyFact))

def exact78417RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact78417RawTermsValid :
    exact78417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78417 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15579⟩⟩) exact78417RawTerms (.finite 10) 78416 .exactZero (none)

def event78418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15580⟩⟩) 0 ⟨15579⟩ 78417

def event78419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.identity (.predecessor 0 78418 .coefficient))

def event78420 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.finite 10)

def event78421 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20892⟩⟩) 0 ⟨15580⟩ 78420

def event78422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20892⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact78423RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩]

theorem exact78423RawTermsValid :
    exact78423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78423 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20892⟩⟩) exact78423RawTerms (.finite 136065468) 78422 .exactZero (none)

def event78424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact78425RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact78425RawTermsValid :
    exact78425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact78425RawTerms .large 78424 .exactZero (none)

def event78426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20893⟩⟩) 0 ⟨6⟩ 78425

def event78427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20893⟩⟩) 1 ⟨20892⟩ 78423

def event78428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20893⟩⟩) (.product (.predecessor 0 78426 .coefficient) (.predecessor 1 78427 .coefficient) (⟨false, false, none, none, none⟩))

def event78429 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20893⟩⟩, .operator (⟨78425, 0⟩, ⟨78423, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩)

def exact78430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩]

theorem exact78430RawTermsValid :
    exact78430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20893⟩⟩) exact78430RawTerms .large 78428 .exactZero (none)

def event78431 : Event := .preFoldPolynomial 78430 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩] .exactZero none

def exact78432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩, (1)⟩]

def event78432 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20893⟩⟩) 78431 exact78432RawTerms .large 78428 .exactZero (none)

def event78433 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27201⟩⟩)

def event78434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78435 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78437 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78441 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78441

def event78443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78439

def event78444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78442 .coefficient) (.value (.predecessor 1 78443 .coefficient)))

def event78445 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78445

def event78447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78437

def event78448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78446 .coefficient, .predecessor 1 78447 .coefficient])

def event78449 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78449

def event78451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78435

def event78452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78451 .coefficient))

def event78453 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11213⟩⟩) 0 ⟨5530⟩ 78453

def event78455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11213⟩⟩) (.authority (.programFamilyFact))

def exact78456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩], []⟩, (1)⟩]

theorem exact78456RawTermsValid :
    exact78456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11213⟩⟩) exact78456RawTerms (.finite 10) 78455 .exactZero (none)

def event78457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13547⟩⟩) 0 ⟨5530⟩ 78453

def event78458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13547⟩⟩) (.authority (.programFamilyFact))

def exact78459RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact78459RawTermsValid :
    exact78459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13547⟩⟩) exact78459RawTerms (.finite 10) 78458 .exactZero (none)

def event78460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 0 ⟨13547⟩ 78459

def event78461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13548⟩⟩) 1 ⟨11213⟩ 78456

def event78462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13548⟩⟩) (.product (.predecessor 0 78460 .coefficient) (.predecessor 1 78461 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78463 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13548⟩⟩, .operator (⟨78459, 0⟩, ⟨78456, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩)

def exact78464RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩, (1)⟩]

theorem exact78464RawTermsValid :
    exact78464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13548⟩⟩) exact78464RawTerms (.finite 100) 78462 .exactZero (none)

def event78465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13549⟩⟩) 0 ⟨13548⟩ 78464

def event78466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.identity (.predecessor 0 78465 .coefficient))

def event78467 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13549⟩⟩) (.finite 100)

def event78468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15579⟩⟩) 0 ⟨13549⟩ 78467

def event78469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15579⟩⟩) (.authority (.programFamilyFact))

def exact78470RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact78470RawTermsValid :
    exact78470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78470 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15579⟩⟩) exact78470RawTerms (.finite 10) 78469 .exactZero (none)

def event78471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15580⟩⟩) 0 ⟨15579⟩ 78470

def event78472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.identity (.predecessor 0 78471 .coefficient))

def event78473 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15580⟩⟩) (.finite 10)

def event78474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23968⟩⟩) 0 ⟨15580⟩ 78473

def event78475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23968⟩⟩) (.authority (.programFamilyFact))

def event78476 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23968⟩⟩) (.finite 3720)

def event78477 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event78478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23969⟩⟩) 0 ⟨6689⟩ 78477

def event78479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23969⟩⟩) 1 ⟨23968⟩ 78476

def event78480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23969⟩⟩) (.authority (.operator))

def exact78481RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩]

theorem exact78481RawTermsValid :
    exact78481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23969⟩⟩) exact78481RawTerms .large 78480 .exactZero (none)

def event78482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27195⟩⟩) 0 ⟨23969⟩ 78481

def event78483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27195⟩⟩) (.authority (.operator))

def exact78484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩]

theorem exact78484RawTermsValid :
    exact78484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27195⟩⟩) exact78484RawTerms (.finite 8192) 78483 .exactZero (none)

def event78485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event78486 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event78487 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15654⟩⟩) 0 ⟨15580⟩ 78473

def event78488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15654⟩⟩) 1 ⟨110⟩ 78486

def event78489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15654⟩⟩) (.sum [.predecessor 0 78487 .coefficient, .predecessor 1 78488 .coefficient])

def event78490 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15654⟩⟩) (.finite 10)

def event78491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15655⟩⟩) 0 ⟨15654⟩ 78490

def event78492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15655⟩⟩) (.identity (.predecessor 0 78491 .coefficient))

def exact78493RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], []⟩, (1)⟩]

theorem exact78493RawTermsValid :
    exact78493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78493 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15655⟩⟩) exact78493RawTerms (.finite 10) 78492 .exactZero (none)

def event78494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact78495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78495RawTermsValid :
    exact78495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78495 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact78495RawTerms .large 78494 .exactZero (none)

def event78496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15656⟩⟩) 0 ⟨6544⟩ 78495

def event78497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15656⟩⟩) 1 ⟨15655⟩ 78493

def event78498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15656⟩⟩) (.product (.predecessor 0 78496 .coefficient) (.predecessor 1 78497 .coefficient) (⟨false, false, none, none, none⟩))

def event78499 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15656⟩⟩, .operator (⟨78495, 0⟩, ⟨78493, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78500RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78500RawTermsValid :
    exact78500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15656⟩⟩) exact78500RawTerms .large 78498 .exactZero (none)

def event78501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 78477

def event78502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact78503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact78503RawTermsValid :
    exact78503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact78503RawTerms .large 78502 .exactZero (none)

def event78504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15657⟩⟩) 0 ⟨6694⟩ 78503

def event78505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15657⟩⟩) 1 ⟨15656⟩ 78500

def event78506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15657⟩⟩) (.sum [.predecessor 0 78504 .coefficient, .predecessor 1 78505 .coefficient])

def exact78507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78507RawTermsValid :
    exact78507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78507 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15657⟩⟩) exact78507RawTerms .large 78506 .exactZero (none)

def event78508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27196⟩⟩) 0 ⟨15657⟩ 78507

def event78509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27196⟩⟩) 1 ⟨27195⟩ 78484

def event78510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27196⟩⟩) (.product (.predecessor 0 78508 .coefficient) (.predecessor 1 78509 .coefficient) (⟨false, false, none, none, none⟩))

def event78511 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27196⟩⟩, .operator (⟨78507, 0⟩, ⟨78484, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩)

def event78512 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27196⟩⟩, .operator (⟨78507, 1⟩, ⟨78484, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩)

def event78513 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27196⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27195⟩⟩) ⟨23969⟩ 78481)

def event78514 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27196⟩⟩, .relation 78513 0, ⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (-1)⟩)

def exact78515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (-1)⟩]

theorem exact78515RawTermsValid :
    exact78515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78515 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27196⟩⟩) exact78515RawTerms .large 78510 .exactZero (none)

def event78516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17806⟩⟩) 0 ⟨15580⟩ 78473

def event78517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17806⟩⟩) (.authority (.programFamilyFact))

def exact78518RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], []⟩, (1)⟩]

theorem exact78518RawTermsValid :
    exact78518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78518 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17806⟩⟩) exact78518RawTerms (.finite 10) 78517 .exactZero (none)

def event78519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17812⟩⟩) 0 ⟨6544⟩ 78495

def event78520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17812⟩⟩) 1 ⟨17806⟩ 78518

def event78521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17812⟩⟩) (.product (.predecessor 0 78519 .coefficient) (.predecessor 1 78520 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78522 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17812⟩⟩, .operator (⟨78495, 0⟩, ⟨78518, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78523RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78523RawTermsValid :
    exact78523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78523 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17812⟩⟩) exact78523RawTerms .large 78521 .exactZero (none)

def event78524 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6716⟩⟩) 0 ⟨6689⟩ 78477

def event78525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6716⟩⟩) (.authority (.operator))

def exact78526RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩]

theorem exact78526RawTermsValid :
    exact78526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78526 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6716⟩⟩) exact78526RawTerms .large 78525 .exactZero (none)

def event78527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17813⟩⟩) 0 ⟨6716⟩ 78526

def event78528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17813⟩⟩) 1 ⟨17812⟩ 78523

def event78529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17813⟩⟩) (.sum [.predecessor 0 78527 .coefficient, .predecessor 1 78528 .coefficient])

def exact78530RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78530RawTermsValid :
    exact78530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17813⟩⟩) exact78530RawTerms .large 78529 .exactZero (none)

def event78531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27201⟩⟩) 0 ⟨17813⟩ 78530

def event78532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27201⟩⟩) 1 ⟨27196⟩ 78515

def event78533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27201⟩⟩) (.sum [.predecessor 0 78531 .coefficient, .predecessor 1 78532 .coefficient])

def exact78534RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78534RawTermsValid :
    exact78534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27201⟩⟩) exact78534RawTerms .large 78533 .exactZero (none)

def event78535 : Event := .preFoldPolynomial 78534 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event78536 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27201⟩⟩) 78535 exact78536RawTerms .large 78533 .exactZero (none)

def event78537 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15580⟩⟩) ⟨⟨129⟩, ⟨36⟩, ⟨109⟩⟩ ⟨78379, 78537⟩

def event78538 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20895⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩) (1) 0 2 (.universal 78537 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20892⟩⟩]⟩) (none) 78536)

def event78539 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20895⟩⟩, .relation 78538 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩)

def event78540 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20895⟩⟩, .relation 78538 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩)

def event78541 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20895⟩⟩, .relation 78538 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩)

def event78542 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20895⟩⟩, .relation 78538 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78543RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78543RawTermsValid :
    exact78543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78543 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20895⟩⟩) exact78543RawTerms .large 78375 (.finite 1811303510016) (some (78377))

def event78544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27198⟩⟩) 0 ⟨20895⟩ 78543

def event78545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27198⟩⟩) 1 ⟨27197⟩ 78365

def event78546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27198⟩⟩) (.sum [.predecessor 0 78544 .coefficient, .predecessor 1 78545 .coefficient])

def event78547 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27198⟩⟩, .operator (⟨78543, 0⟩, ⟨78365, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27195⟩⟩]⟩, (1)⟩)

def event78548 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27198⟩⟩, .operator (⟨78543, 2⟩, ⟨78365, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15579⟩⟩], [⟨.program ⟨214⟩, ⟨23969⟩⟩]⟩, (-1)⟩)

def event78549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27198⟩⟩) (.sum [.result 78543 .summary, .result 78365 .summary])

def exact78550RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78550RawTermsValid :
    exact78550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78550 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27198⟩⟩) exact78550RawTerms .large 78546 (.finite 1291978824159503986688) (some (78549))

def event78551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27199⟩⟩) 0 ⟨27198⟩ 78550

def event78552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27199⟩⟩) 1 ⟨6650⟩ 5779

def event78553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27199⟩⟩) (.product (.predecessor 0 78551 .coefficient) (.predecessor 1 78552 .coefficient) (⟨false, false, none, none, none⟩))

def event78554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27199⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) [⟨.result 5775 .coefficient, false, none⟩])

def event78555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27199⟩⟩) (.product (.result 78550 .summary) (.transfer 78554) (⟨false, false, none, none, none⟩))

def event78556 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27199⟩⟩, .operator (⟨78550, 0⟩, ⟨5779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩)

def event78557 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27199⟩⟩, .operator (⟨78550, 1⟩, ⟨5779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (-1)⟩)

def event78558 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27199⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6649⟩⟩) ⟨6596⟩ 5772)

def event78559 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27199⟩⟩, .relation 78558 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78560RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78560RawTermsValid :
    exact78560RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78560 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27199⟩⟩) exact78560RawTerms .large 78553 (.finite 4741582956326566183208747008) (some (78555))

def event78561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23906⟩⟩) 0 ⟨6689⟩ 5477

def event78562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23906⟩⟩) 1 ⟨23905⟩ 72037

def event78563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23906⟩⟩) (.authority (.operator))

def exact78564RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩]

theorem exact78564RawTermsValid :
    exact78564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78564 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23906⟩⟩) exact78564RawTerms .large 78563 .exactZero (none)

def event78565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26978⟩⟩) 0 ⟨23906⟩ 78564

def event78566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26978⟩⟩) (.authority (.operator))

def exact78567RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩]

theorem exact78567RawTermsValid :
    exact78567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78567 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26978⟩⟩) exact78567RawTerms (.finite 8192) 78566 .exactZero (none)

def event78568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26980⟩⟩) 0 ⟨25293⟩ 72321

def event78569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26980⟩⟩) 1 ⟨26978⟩ 78567

def event78570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26980⟩⟩) (.product (.predecessor 0 78568 .coefficient) (.predecessor 1 78569 .coefficient) (⟨false, false, none, none, none⟩))

def event78571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26980⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩) [⟨.result 78567 .coefficient, false, none⟩])

def event78572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26980⟩⟩) (.product (.result 72321 .summary) (.transfer 78571) (⟨false, false, none, none, none⟩))

def event78573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26980⟩⟩, .operator (⟨72321, 0⟩, ⟨78567, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩)

def event78574 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26980⟩⟩, .operator (⟨72321, 1⟩, ⟨78567, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩)

def event78575 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26980⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26978⟩⟩) ⟨23906⟩ 78564)

def event78576 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26980⟩⟩, .relation 78575 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (-1)⟩)

def exact78577RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (-1)⟩]

theorem exact78577RawTermsValid :
    exact78577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26980⟩⟩) exact78577RawTerms .large 78570 (.finite 1291933997458159304704) (some (78572))

def event78578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20748⟩⟩) 0 ⟨15419⟩ 3425

def event78579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20748⟩⟩) (.authority (.relationPreimageSource ⟨34⟩))

def exact78580RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩]

theorem exact78580RawTermsValid :
    exact78580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20748⟩⟩) exact78580RawTerms (.finite 136065468) 78579 .exactZero (none)

def event78581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20750⟩⟩) 0 ⟨20748⟩ 78580

def event78582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20750⟩⟩) 1 ⟨2348⟩ 4

def event78583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20750⟩⟩) (.scale (.predecessor 0 78581 .coefficient) (.value (.predecessor 1 78582 .coefficient)))

def exact78584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩]

theorem exact78584RawTermsValid :
    exact78584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20750⟩⟩) exact78584RawTerms (.finite 136065468) 78583 .exactZero (none)

def event78585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20751⟩⟩) 0 ⟨5535⟩ 65387

def event78586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20751⟩⟩) 1 ⟨20750⟩ 78584

def event78587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20751⟩⟩) (.product (.predecessor 0 78585 .coefficient) (.predecessor 1 78586 .coefficient) (⟨false, false, none, none, none⟩))

def event78588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20751⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩) [⟨.result 78580 .coefficient, false, none⟩])

def event78589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20751⟩⟩) (.product (.result 65387 .summary) (.transfer 78588) (⟨false, false, none, none, none⟩))

def event78590 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20751⟩⟩, .operator (⟨65387, 0⟩, ⟨78584, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩)

def event78591 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20749⟩⟩)

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
    frameStart := 78379 },
  { event := event78380
    frameStart := 78379 },
  { event := event78381
    frameStart := 78379 },
  { event := event78382
    frameStart := 78379 },
  { event := event78383
    frameStart := 78379 }
]

def eventLeaf4899 : Array AnnotatedEvent := #[
  { event := event78384
    frameStart := 78379 },
  { event := event78385
    frameStart := 78379 },
  { event := event78386
    frameStart := 78379 },
  { event := event78387
    frameStart := 78379 },
  { event := event78388
    frameStart := 78379 },
  { event := event78389
    frameStart := 78379 },
  { event := event78390
    frameStart := 78379 },
  { event := event78391
    frameStart := 78379 },
  { event := event78392
    frameStart := 78379 },
  { event := event78393
    frameStart := 78379 },
  { event := event78394
    frameStart := 78379 },
  { event := event78395
    frameStart := 78379 },
  { event := event78396
    frameStart := 78379 },
  { event := event78397
    frameStart := 78379 },
  { event := event78398
    frameStart := 78379 },
  { event := event78399
    frameStart := 78379 }
]

def eventLeaf4900 : Array AnnotatedEvent := #[
  { event := event78400
    frameStart := 78379 },
  { event := event78401
    frameStart := 78379 },
  { event := event78402
    frameStart := 78379 },
  { event := event78403
    frameStart := 78379 },
  { event := event78404
    frameStart := 78379 },
  { event := event78405
    frameStart := 78379 },
  { event := event78406
    frameStart := 78379 },
  { event := event78407
    frameStart := 78379 },
  { event := event78408
    frameStart := 78379 },
  { event := event78409
    frameStart := 78379 },
  { event := event78410
    frameStart := 78379 },
  { event := event78411
    frameStart := 78379 },
  { event := event78412
    frameStart := 78379 },
  { event := event78413
    frameStart := 78379 },
  { event := event78414
    frameStart := 78379 },
  { event := event78415
    frameStart := 78379 }
]

def eventLeaf4901 : Array AnnotatedEvent := #[
  { event := event78416
    frameStart := 78379 },
  { event := event78417
    frameStart := 78379 },
  { event := event78418
    frameStart := 78379 },
  { event := event78419
    frameStart := 78379 },
  { event := event78420
    frameStart := 78379 },
  { event := event78421
    frameStart := 78379 },
  { event := event78422
    frameStart := 78379 },
  { event := event78423
    frameStart := 78379 },
  { event := event78424
    frameStart := 78379 },
  { event := event78425
    frameStart := 78379 },
  { event := event78426
    frameStart := 78379 },
  { event := event78427
    frameStart := 78379 },
  { event := event78428
    frameStart := 78379 },
  { event := event78429
    frameStart := 78379 },
  { event := event78430
    frameStart := 78379 },
  { event := event78431
    frameStart := 78379 }
]

def eventLeaf4902 : Array AnnotatedEvent := #[
  { event := event78432
    frameStart := 78379 },
  { event := event78433
    frameStart := 78433 },
  { event := event78434
    frameStart := 78433 },
  { event := event78435
    frameStart := 78433 },
  { event := event78436
    frameStart := 78433 },
  { event := event78437
    frameStart := 78433 },
  { event := event78438
    frameStart := 78433 },
  { event := event78439
    frameStart := 78433 },
  { event := event78440
    frameStart := 78433 },
  { event := event78441
    frameStart := 78433 },
  { event := event78442
    frameStart := 78433 },
  { event := event78443
    frameStart := 78433 },
  { event := event78444
    frameStart := 78433 },
  { event := event78445
    frameStart := 78433 },
  { event := event78446
    frameStart := 78433 },
  { event := event78447
    frameStart := 78433 }
]

def eventLeaf4903 : Array AnnotatedEvent := #[
  { event := event78448
    frameStart := 78433 },
  { event := event78449
    frameStart := 78433 },
  { event := event78450
    frameStart := 78433 },
  { event := event78451
    frameStart := 78433 },
  { event := event78452
    frameStart := 78433 },
  { event := event78453
    frameStart := 78433 },
  { event := event78454
    frameStart := 78433 },
  { event := event78455
    frameStart := 78433 },
  { event := event78456
    frameStart := 78433 },
  { event := event78457
    frameStart := 78433 },
  { event := event78458
    frameStart := 78433 },
  { event := event78459
    frameStart := 78433 },
  { event := event78460
    frameStart := 78433 },
  { event := event78461
    frameStart := 78433 },
  { event := event78462
    frameStart := 78433 },
  { event := event78463
    frameStart := 78433 }
]

def eventLeaf4904 : Array AnnotatedEvent := #[
  { event := event78464
    frameStart := 78433 },
  { event := event78465
    frameStart := 78433 },
  { event := event78466
    frameStart := 78433 },
  { event := event78467
    frameStart := 78433 },
  { event := event78468
    frameStart := 78433 },
  { event := event78469
    frameStart := 78433 },
  { event := event78470
    frameStart := 78433 },
  { event := event78471
    frameStart := 78433 },
  { event := event78472
    frameStart := 78433 },
  { event := event78473
    frameStart := 78433 },
  { event := event78474
    frameStart := 78433 },
  { event := event78475
    frameStart := 78433 },
  { event := event78476
    frameStart := 78433 },
  { event := event78477
    frameStart := 78433 },
  { event := event78478
    frameStart := 78433 },
  { event := event78479
    frameStart := 78433 }
]

def eventLeaf4905 : Array AnnotatedEvent := #[
  { event := event78480
    frameStart := 78433 },
  { event := event78481
    frameStart := 78433 },
  { event := event78482
    frameStart := 78433 },
  { event := event78483
    frameStart := 78433 },
  { event := event78484
    frameStart := 78433 },
  { event := event78485
    frameStart := 78433 },
  { event := event78486
    frameStart := 78433 },
  { event := event78487
    frameStart := 78433 },
  { event := event78488
    frameStart := 78433 },
  { event := event78489
    frameStart := 78433 },
  { event := event78490
    frameStart := 78433 },
  { event := event78491
    frameStart := 78433 },
  { event := event78492
    frameStart := 78433 },
  { event := event78493
    frameStart := 78433 },
  { event := event78494
    frameStart := 78433 },
  { event := event78495
    frameStart := 78433 }
]

def eventLeaf4906 : Array AnnotatedEvent := #[
  { event := event78496
    frameStart := 78433 },
  { event := event78497
    frameStart := 78433 },
  { event := event78498
    frameStart := 78433 },
  { event := event78499
    frameStart := 78433 },
  { event := event78500
    frameStart := 78433 },
  { event := event78501
    frameStart := 78433 },
  { event := event78502
    frameStart := 78433 },
  { event := event78503
    frameStart := 78433 },
  { event := event78504
    frameStart := 78433 },
  { event := event78505
    frameStart := 78433 },
  { event := event78506
    frameStart := 78433 },
  { event := event78507
    frameStart := 78433 },
  { event := event78508
    frameStart := 78433 },
  { event := event78509
    frameStart := 78433 },
  { event := event78510
    frameStart := 78433 },
  { event := event78511
    frameStart := 78433 }
]

def eventLeaf4907 : Array AnnotatedEvent := #[
  { event := event78512
    frameStart := 78433 },
  { event := event78513
    frameStart := 78433 },
  { event := event78514
    frameStart := 78433 },
  { event := event78515
    frameStart := 78433 },
  { event := event78516
    frameStart := 78433 },
  { event := event78517
    frameStart := 78433 },
  { event := event78518
    frameStart := 78433 },
  { event := event78519
    frameStart := 78433 },
  { event := event78520
    frameStart := 78433 },
  { event := event78521
    frameStart := 78433 },
  { event := event78522
    frameStart := 78433 },
  { event := event78523
    frameStart := 78433 },
  { event := event78524
    frameStart := 78433 },
  { event := event78525
    frameStart := 78433 },
  { event := event78526
    frameStart := 78433 },
  { event := event78527
    frameStart := 78433 }
]

def eventLeaf4908 : Array AnnotatedEvent := #[
  { event := event78528
    frameStart := 78433 },
  { event := event78529
    frameStart := 78433 },
  { event := event78530
    frameStart := 78433 },
  { event := event78531
    frameStart := 78433 },
  { event := event78532
    frameStart := 78433 },
  { event := event78533
    frameStart := 78433 },
  { event := event78534
    frameStart := 78433 },
  { event := event78535
    frameStart := 78433 },
  { event := event78536
    frameStart := 78433 },
  { event := event78537
    frameStart := 0 },
  { event := event78538
    frameStart := 0 },
  { event := event78539
    frameStart := 0 },
  { event := event78540
    frameStart := 0 },
  { event := event78541
    frameStart := 0 },
  { event := event78542
    frameStart := 0 },
  { event := event78543
    frameStart := 0 }
]

def eventLeaf4909 : Array AnnotatedEvent := #[
  { event := event78544
    frameStart := 0 },
  { event := event78545
    frameStart := 0 },
  { event := event78546
    frameStart := 0 },
  { event := event78547
    frameStart := 0 },
  { event := event78548
    frameStart := 0 },
  { event := event78549
    frameStart := 0 },
  { event := event78550
    frameStart := 0 },
  { event := event78551
    frameStart := 0 },
  { event := event78552
    frameStart := 0 },
  { event := event78553
    frameStart := 0 },
  { event := event78554
    frameStart := 0 },
  { event := event78555
    frameStart := 0 },
  { event := event78556
    frameStart := 0 },
  { event := event78557
    frameStart := 0 },
  { event := event78558
    frameStart := 0 },
  { event := event78559
    frameStart := 0 }
]

def eventLeaf4910 : Array AnnotatedEvent := #[
  { event := event78560
    frameStart := 0 },
  { event := event78561
    frameStart := 0 },
  { event := event78562
    frameStart := 0 },
  { event := event78563
    frameStart := 0 },
  { event := event78564
    frameStart := 0 },
  { event := event78565
    frameStart := 0 },
  { event := event78566
    frameStart := 0 },
  { event := event78567
    frameStart := 0 },
  { event := event78568
    frameStart := 0 },
  { event := event78569
    frameStart := 0 },
  { event := event78570
    frameStart := 0 },
  { event := event78571
    frameStart := 0 },
  { event := event78572
    frameStart := 0 },
  { event := event78573
    frameStart := 0 },
  { event := event78574
    frameStart := 0 },
  { event := event78575
    frameStart := 0 }
]

def eventLeaf4911 : Array AnnotatedEvent := #[
  { event := event78576
    frameStart := 0 },
  { event := event78577
    frameStart := 0 },
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
    frameStart := 78591 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events306
