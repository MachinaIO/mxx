import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events185

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact47360RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩, (1)⟩]

theorem exact47360RawTermsValid :
    exact47360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47360 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17957⟩⟩) exact47360RawTerms (.finite 42) 47359 .exactZero (none)

def event47361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17959⟩⟩) 0 ⟨6544⟩ 47337

def event47362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17959⟩⟩) 1 ⟨17957⟩ 47360

def event47363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17959⟩⟩) (.product (.predecessor 0 47361 .coefficient) (.predecessor 1 47362 .coefficient) (⟨false, true, none, none, some 1⟩))

def event47364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17959⟩⟩, .operator (⟨47337, 0⟩, ⟨47360, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact47365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47365RawTermsValid :
    exact47365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17959⟩⟩) exact47365RawTerms .large 47363 .exactZero (none)

def event47366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6734⟩⟩) 0 ⟨6689⟩ 47319

def event47367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6734⟩⟩) (.authority (.operator))

def exact47368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩]

theorem exact47368RawTermsValid :
    exact47368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6734⟩⟩) exact47368RawTerms .large 47367 .exactZero (none)

def event47369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17960⟩⟩) 0 ⟨6734⟩ 47368

def event47370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17960⟩⟩) 1 ⟨17959⟩ 47365

def event47371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17960⟩⟩) (.sum [.predecessor 0 47369 .coefficient, .predecessor 1 47370 .coefficient])

def exact47372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47372RawTermsValid :
    exact47372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17960⟩⟩) exact47372RawTerms .large 47371 .exactZero (none)

def event47373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29193⟩⟩) 0 ⟨17960⟩ 47372

def event47374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29193⟩⟩) 1 ⟨29188⟩ 47357

def event47375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29193⟩⟩) (.sum [.predecessor 0 47373 .coefficient, .predecessor 1 47374 .coefficient])

def exact47376RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47376RawTermsValid :
    exact47376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29193⟩⟩) exact47376RawTerms .large 47375 .exactZero (none)

def event47377 : Event := .preFoldPolynomial 47376 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact47378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event47378 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29193⟩⟩) 47377 exact47378RawTerms .large 47375 .exactZero (none)

def event47379 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16558⟩⟩) ⟨⟨147⟩, ⟨56⟩, ⟨109⟩⟩ ⟨47221, 47379⟩

def event47380 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22203⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22200⟩⟩]⟩) (1) 0 2 (.universal 47379 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22200⟩⟩]⟩) (none) 47378)

def event47381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22203⟩⟩, .relation 47380 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩)

def event47382 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22203⟩⟩, .relation 47380 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (-1)⟩)

def event47383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22203⟩⟩, .relation 47380 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (1)⟩)

def event47384 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22203⟩⟩, .relation 47380 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact47385RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47385RawTermsValid :
    exact47385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22203⟩⟩) exact47385RawTerms .large 47217 (.finite 1811303510016) (some (47219))

def event47386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29190⟩⟩) 0 ⟨22203⟩ 47385

def event47387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29190⟩⟩) 1 ⟨29189⟩ 47207

def event47388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29190⟩⟩) (.sum [.predecessor 0 47386 .coefficient, .predecessor 1 47387 .coefficient])

def event47389 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29190⟩⟩, .operator (⟨47385, 0⟩, ⟨47207, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29187⟩⟩]⟩, (1)⟩)

def event47390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29190⟩⟩, .operator (⟨47385, 2⟩, ⟨47207, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16557⟩⟩], [⟨.program ⟨214⟩, ⟨24545⟩⟩]⟩, (-1)⟩)

def event47391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29190⟩⟩) (.sum [.result 47385 .summary, .result 47207 .summary])

def exact47392RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47392RawTermsValid :
    exact47392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47392 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29190⟩⟩) exact47392RawTerms .large 47388 (.finite 1292337423279833362432) (some (47391))

def event47393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29191⟩⟩) 0 ⟨29190⟩ 47392

def event47394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29191⟩⟩) 1 ⟨6668⟩ 5599

def event47395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29191⟩⟩) (.product (.predecessor 0 47393 .coefficient) (.predecessor 1 47394 .coefficient) (⟨false, false, none, none, none⟩))

def event47396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29191⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩) [⟨.result 5595 .coefficient, false, none⟩])

def event47397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29191⟩⟩) (.product (.result 47392 .summary) (.transfer 47396) (⟨false, false, none, none, none⟩))

def event47398 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29191⟩⟩, .operator (⟨47392, 0⟩, ⟨5599, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩)

def event47399 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29191⟩⟩, .operator (⟨47392, 1⟩, ⟨5599, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (-1)⟩)

def event47400 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29191⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6667⟩⟩) ⟨6605⟩ 5592)

def event47401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29191⟩⟩, .relation 47400 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact47402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47402RawTermsValid :
    exact47402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29191⟩⟩) exact47402RawTerms .large 47395 (.finite 4742899020835760917459238912) (some (47397))

def event47403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24482⟩⟩) 0 ⟨6689⟩ 5477

def event47404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24482⟩⟩) 1 ⟨24481⟩ 38449

def event47405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24482⟩⟩) (.authority (.operator))

def exact47406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩]

theorem exact47406RawTermsValid :
    exact47406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24482⟩⟩) exact47406RawTerms .large 47405 .exactZero (none)

def event47407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28970⟩⟩) 0 ⟨24482⟩ 47406

def event47408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28970⟩⟩) (.authority (.operator))

def exact47409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩]

theorem exact47409RawTermsValid :
    exact47409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28970⟩⟩) exact47409RawTerms (.finite 8192) 47408 .exactZero (none)

def event47410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28972⟩⟩) 0 ⟨25385⟩ 38733

def event47411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28972⟩⟩) 1 ⟨28970⟩ 47409

def event47412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28972⟩⟩) (.product (.predecessor 0 47410 .coefficient) (.predecessor 1 47411 .coefficient) (⟨false, false, none, none, none⟩))

def event47413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28972⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩) [⟨.result 47409 .coefficient, false, none⟩])

def event47414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28972⟩⟩) (.product (.result 38733 .summary) (.transfer 47413) (⟨false, false, none, none, none⟩))

def event47415 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28972⟩⟩, .operator (⟨38733, 0⟩, ⟨47409, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩)

def event47416 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28972⟩⟩, .operator (⟨38733, 1⟩, ⟨47409, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩)

def event47417 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28972⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28970⟩⟩) ⟨24482⟩ 47406)

def event47418 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28972⟩⟩, .relation 47417 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (-1)⟩)

def exact47419RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (-1)⟩]

theorem exact47419RawTermsValid :
    exact47419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28972⟩⟩) exact47419RawTerms .large 47412 (.finite 1292315009023509266432) (some (47414))

def event47420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22056⟩⟩) 0 ⟨16474⟩ 1722

def event47421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22056⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact47422RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩]

theorem exact47422RawTermsValid :
    exact47422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22056⟩⟩) exact47422RawTerms (.finite 136065468) 47421 .exactZero (none)

def event47423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22058⟩⟩) 0 ⟨22056⟩ 47422

def event47424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22058⟩⟩) 1 ⟨2348⟩ 4

def event47425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22058⟩⟩) (.scale (.predecessor 0 47423 .coefficient) (.value (.predecessor 1 47424 .coefficient)))

def exact47426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩]

theorem exact47426RawTermsValid :
    exact47426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22058⟩⟩) exact47426RawTerms (.finite 136065468) 47425 .exactZero (none)

def event47427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22059⟩⟩) 0 ⟨5553⟩ 36137

def event47428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22059⟩⟩) 1 ⟨22058⟩ 47426

def event47429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22059⟩⟩) (.product (.predecessor 0 47427 .coefficient) (.predecessor 1 47428 .coefficient) (⟨false, false, none, none, none⟩))

def event47430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22059⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩) [⟨.result 47422 .coefficient, false, none⟩])

def event47431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22059⟩⟩) (.product (.result 36137 .summary) (.transfer 47430) (⟨false, false, none, none, none⟩))

def event47432 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22059⟩⟩, .operator (⟨36137, 0⟩, ⟨47426, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩)

def event47433 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22057⟩⟩)

def event47434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event47435 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event47436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event47437 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event47438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event47439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event47440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event47441 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event47442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 47441

def event47443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 47439

def event47444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 47442 .coefficient) (.value (.predecessor 1 47443 .coefficient)))

def event47445 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event47446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 47445

def event47447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 47437

def event47448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 47446 .coefficient, .predecessor 1 47447 .coefficient])

def event47449 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event47450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 47449

def event47451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 47435

def event47452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 47451 .coefficient))

def event47453 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event47454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12386⟩⟩) 0 ⟨5548⟩ 47453

def event47455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12386⟩⟩) (.authority (.programFamilyFact))

def exact47456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩, (1)⟩]

theorem exact47456RawTermsValid :
    exact47456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12386⟩⟩) exact47456RawTerms (.finite 40) 47455 .exactZero (none)

def event47457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9830⟩⟩) 0 ⟨5548⟩ 47453

def event47458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9830⟩⟩) (.authority (.programFamilyFact))

def exact47459RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩], []⟩, (1)⟩]

theorem exact47459RawTermsValid :
    exact47459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9830⟩⟩) exact47459RawTerms (.finite 40) 47458 .exactZero (none)

def event47460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12387⟩⟩) 0 ⟨9830⟩ 47459

def event47461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12387⟩⟩) 1 ⟨12386⟩ 47456

def event47462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12387⟩⟩) (.product (.predecessor 0 47460 .coefficient) (.predecessor 1 47461 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12387⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩) [⟨.result 47459 .coefficient, true, some 1⟩, ⟨.result 47456 .coefficient, true, some 1⟩])

def event47464 : Event := .survivorFold (1) 47463

def exact47465RawTerms : List Term := []

theorem exact47465RawTermsValid :
    exact47465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47465 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12387⟩⟩) exact47465RawTerms (.finite 1600) 47462 (.finite 1600) (some (47463))

def event47466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12388⟩⟩) 0 ⟨12387⟩ 47465

def event47467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12388⟩⟩) (.identity (.predecessor 0 47466 .coefficient))

def event47468 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12388⟩⟩) (.finite 1600)

def event47469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16473⟩⟩) 0 ⟨12388⟩ 47468

def event47470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16473⟩⟩) (.authority (.programFamilyFact))

def exact47471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], []⟩, (1)⟩]

theorem exact47471RawTermsValid :
    exact47471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16473⟩⟩) exact47471RawTerms (.finite 40) 47470 .exactZero (none)

def event47472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16474⟩⟩) 0 ⟨16473⟩ 47471

def event47473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16474⟩⟩) (.identity (.predecessor 0 47472 .coefficient))

def event47474 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16474⟩⟩) (.finite 40)

def event47475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22056⟩⟩) 0 ⟨16474⟩ 47474

def event47476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22056⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact47477RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩]

theorem exact47477RawTermsValid :
    exact47477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47477 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22056⟩⟩) exact47477RawTerms (.finite 136065468) 47476 .exactZero (none)

def event47478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact47479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact47479RawTermsValid :
    exact47479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact47479RawTerms .large 47478 .exactZero (none)

def event47480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22057⟩⟩) 0 ⟨6⟩ 47479

def event47481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22057⟩⟩) 1 ⟨22056⟩ 47477

def event47482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22057⟩⟩) (.product (.predecessor 0 47480 .coefficient) (.predecessor 1 47481 .coefficient) (⟨false, false, none, none, none⟩))

def event47483 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22057⟩⟩, .operator (⟨47479, 0⟩, ⟨47477, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩)

def exact47484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩]

theorem exact47484RawTermsValid :
    exact47484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22057⟩⟩) exact47484RawTerms .large 47482 .exactZero (none)

def event47485 : Event := .preFoldPolynomial 47484 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩] .exactZero none

def exact47486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩, (1)⟩]

def event47486 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22057⟩⟩) 47485 exact47486RawTerms .large 47482 .exactZero (none)

def event47487 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28976⟩⟩)

def event47488 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event47489 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event47490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event47491 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event47492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event47493 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event47494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event47495 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event47496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 47495

def event47497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 47493

def event47498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 47496 .coefficient) (.value (.predecessor 1 47497 .coefficient)))

def event47499 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event47500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 47499

def event47501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 47491

def event47502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 47500 .coefficient, .predecessor 1 47501 .coefficient])

def event47503 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event47504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 47503

def event47505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 47489

def event47506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 47505 .coefficient))

def event47507 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event47508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12386⟩⟩) 0 ⟨5548⟩ 47507

def event47509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12386⟩⟩) (.authority (.programFamilyFact))

def exact47510RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩, (1)⟩]

theorem exact47510RawTermsValid :
    exact47510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47510 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12386⟩⟩) exact47510RawTerms (.finite 40) 47509 .exactZero (none)

def event47511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9830⟩⟩) 0 ⟨5548⟩ 47507

def event47512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9830⟩⟩) (.authority (.programFamilyFact))

def exact47513RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩], []⟩, (1)⟩]

theorem exact47513RawTermsValid :
    exact47513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47513 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9830⟩⟩) exact47513RawTerms (.finite 40) 47512 .exactZero (none)

def event47514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12387⟩⟩) 0 ⟨9830⟩ 47513

def event47515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12387⟩⟩) 1 ⟨12386⟩ 47510

def event47516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12387⟩⟩) (.product (.predecessor 0 47514 .coefficient) (.predecessor 1 47515 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47517 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12387⟩⟩, .operator (⟨47513, 0⟩, ⟨47510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩, (1)⟩)

def exact47518RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩, (1)⟩]

theorem exact47518RawTermsValid :
    exact47518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47518 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12387⟩⟩) exact47518RawTerms (.finite 1600) 47516 .exactZero (none)

def event47519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12388⟩⟩) 0 ⟨12387⟩ 47518

def event47520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12388⟩⟩) (.identity (.predecessor 0 47519 .coefficient))

def event47521 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12388⟩⟩) (.finite 1600)

def event47522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16473⟩⟩) 0 ⟨12388⟩ 47521

def event47523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16473⟩⟩) (.authority (.programFamilyFact))

def exact47524RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], []⟩, (1)⟩]

theorem exact47524RawTermsValid :
    exact47524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47524 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16473⟩⟩) exact47524RawTerms (.finite 40) 47523 .exactZero (none)

def event47525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16474⟩⟩) 0 ⟨16473⟩ 47524

def event47526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16474⟩⟩) (.identity (.predecessor 0 47525 .coefficient))

def event47527 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16474⟩⟩) (.finite 40)

def event47528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24481⟩⟩) 0 ⟨16474⟩ 47527

def event47529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24481⟩⟩) (.authority (.programFamilyFact))

def event47530 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24481⟩⟩) (.finite 3720)

def event47531 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event47532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24482⟩⟩) 0 ⟨6689⟩ 47531

def event47533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24482⟩⟩) 1 ⟨24481⟩ 47530

def event47534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24482⟩⟩) (.authority (.operator))

def exact47535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩]

theorem exact47535RawTermsValid :
    exact47535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24482⟩⟩) exact47535RawTerms .large 47534 .exactZero (none)

def event47536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28970⟩⟩) 0 ⟨24482⟩ 47535

def event47537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28970⟩⟩) (.authority (.operator))

def exact47538RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩]

theorem exact47538RawTermsValid :
    exact47538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28970⟩⟩) exact47538RawTerms (.finite 8192) 47537 .exactZero (none)

def event47539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event47540 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event47541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16513⟩⟩) 0 ⟨16474⟩ 47527

def event47542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16513⟩⟩) 1 ⟨110⟩ 47540

def event47543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16513⟩⟩) (.sum [.predecessor 0 47541 .coefficient, .predecessor 1 47542 .coefficient])

def event47544 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16513⟩⟩) (.finite 40)

def event47545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16514⟩⟩) 0 ⟨16513⟩ 47544

def event47546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16514⟩⟩) (.identity (.predecessor 0 47545 .coefficient))

def exact47547RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], []⟩, (1)⟩]

theorem exact47547RawTermsValid :
    exact47547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47547 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16514⟩⟩) exact47547RawTerms (.finite 40) 47546 .exactZero (none)

def event47548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact47549RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47549RawTermsValid :
    exact47549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact47549RawTerms .large 47548 .exactZero (none)

def event47550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16515⟩⟩) 0 ⟨6544⟩ 47549

def event47551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16515⟩⟩) 1 ⟨16514⟩ 47547

def event47552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16515⟩⟩) (.product (.predecessor 0 47550 .coefficient) (.predecessor 1 47551 .coefficient) (⟨false, false, none, none, none⟩))

def event47553 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16515⟩⟩, .operator (⟨47549, 0⟩, ⟨47547, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact47554RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47554RawTermsValid :
    exact47554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47554 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16515⟩⟩) exact47554RawTerms .large 47552 .exactZero (none)

def event47555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 47531

def event47556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact47557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact47557RawTermsValid :
    exact47557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact47557RawTerms .large 47556 .exactZero (none)

def event47558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16516⟩⟩) 0 ⟨6702⟩ 47557

def event47559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16516⟩⟩) 1 ⟨16515⟩ 47554

def event47560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16516⟩⟩) (.sum [.predecessor 0 47558 .coefficient, .predecessor 1 47559 .coefficient])

def exact47561RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47561RawTermsValid :
    exact47561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16516⟩⟩) exact47561RawTerms .large 47560 .exactZero (none)

def event47562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28971⟩⟩) 0 ⟨16516⟩ 47561

def event47563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28971⟩⟩) 1 ⟨28970⟩ 47538

def event47564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28971⟩⟩) (.product (.predecessor 0 47562 .coefficient) (.predecessor 1 47563 .coefficient) (⟨false, false, none, none, none⟩))

def event47565 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28971⟩⟩, .operator (⟨47561, 0⟩, ⟨47538, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩)

def event47566 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28971⟩⟩, .operator (⟨47561, 1⟩, ⟨47538, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩)

def event47567 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28971⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28970⟩⟩) ⟨24482⟩ 47535)

def event47568 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28971⟩⟩, .relation 47567 0, ⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (-1)⟩)

def exact47569RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (-1)⟩]

theorem exact47569RawTermsValid :
    exact47569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28971⟩⟩) exact47569RawTerms .large 47564 .exactZero (none)

def event47570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17558⟩⟩) 0 ⟨16474⟩ 47527

def event47571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17558⟩⟩) (.authority (.programFamilyFact))

def exact47572RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩, (1)⟩]

theorem exact47572RawTermsValid :
    exact47572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47572 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17558⟩⟩) exact47572RawTerms (.finite 40) 47571 .exactZero (none)

def event47573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17560⟩⟩) 0 ⟨6544⟩ 47549

def event47574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17560⟩⟩) 1 ⟨17558⟩ 47572

def event47575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17560⟩⟩) (.product (.predecessor 0 47573 .coefficient) (.predecessor 1 47574 .coefficient) (⟨false, true, none, none, some 1⟩))

def event47576 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17560⟩⟩, .operator (⟨47549, 0⟩, ⟨47572, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact47577RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47577RawTermsValid :
    exact47577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17560⟩⟩) exact47577RawTerms .large 47575 .exactZero (none)

def event47578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6732⟩⟩) 0 ⟨6689⟩ 47531

def event47579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6732⟩⟩) (.authority (.operator))

def exact47580RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩]

theorem exact47580RawTermsValid :
    exact47580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6732⟩⟩) exact47580RawTerms .large 47579 .exactZero (none)

def event47581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17561⟩⟩) 0 ⟨6732⟩ 47580

def event47582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17561⟩⟩) 1 ⟨17560⟩ 47577

def event47583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17561⟩⟩) (.sum [.predecessor 0 47581 .coefficient, .predecessor 1 47582 .coefficient])

def exact47584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47584RawTermsValid :
    exact47584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17561⟩⟩) exact47584RawTerms .large 47583 .exactZero (none)

def event47585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28976⟩⟩) 0 ⟨17561⟩ 47584

def event47586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28976⟩⟩) 1 ⟨28971⟩ 47569

def event47587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28976⟩⟩) (.sum [.predecessor 0 47585 .coefficient, .predecessor 1 47586 .coefficient])

def exact47588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47588RawTermsValid :
    exact47588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28976⟩⟩) exact47588RawTerms .large 47587 .exactZero (none)

def event47589 : Event := .preFoldPolynomial 47588 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact47590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event47590 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28976⟩⟩) 47589 exact47590RawTerms .large 47587 .exactZero (none)

def event47591 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16474⟩⟩) ⟨⟨145⟩, ⟨53⟩, ⟨109⟩⟩ ⟨47433, 47591⟩

def event47592 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22059⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩) (1) 0 2 (.universal 47591 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩) (none) 47590)

def event47593 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22059⟩⟩, .relation 47592 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩)

def event47594 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22059⟩⟩, .relation 47592 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩)

def event47595 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22059⟩⟩, .relation 47592 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩)

def event47596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22059⟩⟩, .relation 47592 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact47597RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47597RawTermsValid :
    exact47597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22059⟩⟩) exact47597RawTerms .large 47429 (.finite 1811303510016) (some (47431))

def event47598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28973⟩⟩) 0 ⟨22059⟩ 47597

def event47599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28973⟩⟩) 1 ⟨28972⟩ 47419

def event47600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28973⟩⟩) (.sum [.predecessor 0 47598 .coefficient, .predecessor 1 47599 .coefficient])

def event47601 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28973⟩⟩, .operator (⟨47597, 0⟩, ⟨47419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩, (1)⟩)

def event47602 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28973⟩⟩, .operator (⟨47597, 2⟩, ⟨47419, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨24482⟩⟩]⟩, (-1)⟩)

def event47603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28973⟩⟩) (.sum [.result 47597 .summary, .result 47419 .summary])

def exact47604RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47604RawTermsValid :
    exact47604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47604 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28973⟩⟩) exact47604RawTerms .large 47600 (.finite 1292315010834812776448) (some (47603))

def event47605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28974⟩⟩) 0 ⟨28973⟩ 47604

def event47606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28974⟩⟩) 1 ⟨6670⟩ 5619

def event47607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28974⟩⟩) (.product (.predecessor 0 47605 .coefficient) (.predecessor 1 47606 .coefficient) (⟨false, false, none, none, none⟩))

def event47608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28974⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) [⟨.result 5615 .coefficient, false, none⟩])

def event47609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28974⟩⟩) (.product (.result 47604 .summary) (.transfer 47608) (⟨false, false, none, none, none⟩))

def event47610 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28974⟩⟩, .operator (⟨47604, 0⟩, ⟨5619, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩)

def event47611 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28974⟩⟩, .operator (⟨47604, 1⟩, ⟨5619, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (-1)⟩)

def event47612 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28974⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6669⟩⟩) ⟨6606⟩ 5612)

def event47613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28974⟩⟩, .relation 47612 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact47614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47614RawTermsValid :
    exact47614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28974⟩⟩) exact47614RawTerms .large 47607 (.finite 4742816766803936246568583168) (some (47609))

def event47615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24419⟩⟩) 0 ⟨6689⟩ 5477

def eventLeaf2960 : Array AnnotatedEvent := #[
  { event := event47360
    frameStart := 47275 },
  { event := event47361
    frameStart := 47275 },
  { event := event47362
    frameStart := 47275 },
  { event := event47363
    frameStart := 47275 },
  { event := event47364
    frameStart := 47275 },
  { event := event47365
    frameStart := 47275 },
  { event := event47366
    frameStart := 47275 },
  { event := event47367
    frameStart := 47275 },
  { event := event47368
    frameStart := 47275 },
  { event := event47369
    frameStart := 47275 },
  { event := event47370
    frameStart := 47275 },
  { event := event47371
    frameStart := 47275 },
  { event := event47372
    frameStart := 47275 },
  { event := event47373
    frameStart := 47275 },
  { event := event47374
    frameStart := 47275 },
  { event := event47375
    frameStart := 47275 }
]

def eventLeaf2961 : Array AnnotatedEvent := #[
  { event := event47376
    frameStart := 47275 },
  { event := event47377
    frameStart := 47275 },
  { event := event47378
    frameStart := 47275 },
  { event := event47379
    frameStart := 0 },
  { event := event47380
    frameStart := 0 },
  { event := event47381
    frameStart := 0 },
  { event := event47382
    frameStart := 0 },
  { event := event47383
    frameStart := 0 },
  { event := event47384
    frameStart := 0 },
  { event := event47385
    frameStart := 0 },
  { event := event47386
    frameStart := 0 },
  { event := event47387
    frameStart := 0 },
  { event := event47388
    frameStart := 0 },
  { event := event47389
    frameStart := 0 },
  { event := event47390
    frameStart := 0 },
  { event := event47391
    frameStart := 0 }
]

def eventLeaf2962 : Array AnnotatedEvent := #[
  { event := event47392
    frameStart := 0 },
  { event := event47393
    frameStart := 0 },
  { event := event47394
    frameStart := 0 },
  { event := event47395
    frameStart := 0 },
  { event := event47396
    frameStart := 0 },
  { event := event47397
    frameStart := 0 },
  { event := event47398
    frameStart := 0 },
  { event := event47399
    frameStart := 0 },
  { event := event47400
    frameStart := 0 },
  { event := event47401
    frameStart := 0 },
  { event := event47402
    frameStart := 0 },
  { event := event47403
    frameStart := 0 },
  { event := event47404
    frameStart := 0 },
  { event := event47405
    frameStart := 0 },
  { event := event47406
    frameStart := 0 },
  { event := event47407
    frameStart := 0 }
]

def eventLeaf2963 : Array AnnotatedEvent := #[
  { event := event47408
    frameStart := 0 },
  { event := event47409
    frameStart := 0 },
  { event := event47410
    frameStart := 0 },
  { event := event47411
    frameStart := 0 },
  { event := event47412
    frameStart := 0 },
  { event := event47413
    frameStart := 0 },
  { event := event47414
    frameStart := 0 },
  { event := event47415
    frameStart := 0 },
  { event := event47416
    frameStart := 0 },
  { event := event47417
    frameStart := 0 },
  { event := event47418
    frameStart := 0 },
  { event := event47419
    frameStart := 0 },
  { event := event47420
    frameStart := 0 },
  { event := event47421
    frameStart := 0 },
  { event := event47422
    frameStart := 0 },
  { event := event47423
    frameStart := 0 }
]

def eventLeaf2964 : Array AnnotatedEvent := #[
  { event := event47424
    frameStart := 0 },
  { event := event47425
    frameStart := 0 },
  { event := event47426
    frameStart := 0 },
  { event := event47427
    frameStart := 0 },
  { event := event47428
    frameStart := 0 },
  { event := event47429
    frameStart := 0 },
  { event := event47430
    frameStart := 0 },
  { event := event47431
    frameStart := 0 },
  { event := event47432
    frameStart := 0 },
  { event := event47433
    frameStart := 47433 },
  { event := event47434
    frameStart := 47433 },
  { event := event47435
    frameStart := 47433 },
  { event := event47436
    frameStart := 47433 },
  { event := event47437
    frameStart := 47433 },
  { event := event47438
    frameStart := 47433 },
  { event := event47439
    frameStart := 47433 }
]

def eventLeaf2965 : Array AnnotatedEvent := #[
  { event := event47440
    frameStart := 47433 },
  { event := event47441
    frameStart := 47433 },
  { event := event47442
    frameStart := 47433 },
  { event := event47443
    frameStart := 47433 },
  { event := event47444
    frameStart := 47433 },
  { event := event47445
    frameStart := 47433 },
  { event := event47446
    frameStart := 47433 },
  { event := event47447
    frameStart := 47433 },
  { event := event47448
    frameStart := 47433 },
  { event := event47449
    frameStart := 47433 },
  { event := event47450
    frameStart := 47433 },
  { event := event47451
    frameStart := 47433 },
  { event := event47452
    frameStart := 47433 },
  { event := event47453
    frameStart := 47433 },
  { event := event47454
    frameStart := 47433 },
  { event := event47455
    frameStart := 47433 }
]

def eventLeaf2966 : Array AnnotatedEvent := #[
  { event := event47456
    frameStart := 47433 },
  { event := event47457
    frameStart := 47433 },
  { event := event47458
    frameStart := 47433 },
  { event := event47459
    frameStart := 47433 },
  { event := event47460
    frameStart := 47433 },
  { event := event47461
    frameStart := 47433 },
  { event := event47462
    frameStart := 47433 },
  { event := event47463
    frameStart := 47433 },
  { event := event47464
    frameStart := 47433 },
  { event := event47465
    frameStart := 47433 },
  { event := event47466
    frameStart := 47433 },
  { event := event47467
    frameStart := 47433 },
  { event := event47468
    frameStart := 47433 },
  { event := event47469
    frameStart := 47433 },
  { event := event47470
    frameStart := 47433 },
  { event := event47471
    frameStart := 47433 }
]

def eventLeaf2967 : Array AnnotatedEvent := #[
  { event := event47472
    frameStart := 47433 },
  { event := event47473
    frameStart := 47433 },
  { event := event47474
    frameStart := 47433 },
  { event := event47475
    frameStart := 47433 },
  { event := event47476
    frameStart := 47433 },
  { event := event47477
    frameStart := 47433 },
  { event := event47478
    frameStart := 47433 },
  { event := event47479
    frameStart := 47433 },
  { event := event47480
    frameStart := 47433 },
  { event := event47481
    frameStart := 47433 },
  { event := event47482
    frameStart := 47433 },
  { event := event47483
    frameStart := 47433 },
  { event := event47484
    frameStart := 47433 },
  { event := event47485
    frameStart := 47433 },
  { event := event47486
    frameStart := 47433 },
  { event := event47487
    frameStart := 47487 }
]

def eventLeaf2968 : Array AnnotatedEvent := #[
  { event := event47488
    frameStart := 47487 },
  { event := event47489
    frameStart := 47487 },
  { event := event47490
    frameStart := 47487 },
  { event := event47491
    frameStart := 47487 },
  { event := event47492
    frameStart := 47487 },
  { event := event47493
    frameStart := 47487 },
  { event := event47494
    frameStart := 47487 },
  { event := event47495
    frameStart := 47487 },
  { event := event47496
    frameStart := 47487 },
  { event := event47497
    frameStart := 47487 },
  { event := event47498
    frameStart := 47487 },
  { event := event47499
    frameStart := 47487 },
  { event := event47500
    frameStart := 47487 },
  { event := event47501
    frameStart := 47487 },
  { event := event47502
    frameStart := 47487 },
  { event := event47503
    frameStart := 47487 }
]

def eventLeaf2969 : Array AnnotatedEvent := #[
  { event := event47504
    frameStart := 47487 },
  { event := event47505
    frameStart := 47487 },
  { event := event47506
    frameStart := 47487 },
  { event := event47507
    frameStart := 47487 },
  { event := event47508
    frameStart := 47487 },
  { event := event47509
    frameStart := 47487 },
  { event := event47510
    frameStart := 47487 },
  { event := event47511
    frameStart := 47487 },
  { event := event47512
    frameStart := 47487 },
  { event := event47513
    frameStart := 47487 },
  { event := event47514
    frameStart := 47487 },
  { event := event47515
    frameStart := 47487 },
  { event := event47516
    frameStart := 47487 },
  { event := event47517
    frameStart := 47487 },
  { event := event47518
    frameStart := 47487 },
  { event := event47519
    frameStart := 47487 }
]

def eventLeaf2970 : Array AnnotatedEvent := #[
  { event := event47520
    frameStart := 47487 },
  { event := event47521
    frameStart := 47487 },
  { event := event47522
    frameStart := 47487 },
  { event := event47523
    frameStart := 47487 },
  { event := event47524
    frameStart := 47487 },
  { event := event47525
    frameStart := 47487 },
  { event := event47526
    frameStart := 47487 },
  { event := event47527
    frameStart := 47487 },
  { event := event47528
    frameStart := 47487 },
  { event := event47529
    frameStart := 47487 },
  { event := event47530
    frameStart := 47487 },
  { event := event47531
    frameStart := 47487 },
  { event := event47532
    frameStart := 47487 },
  { event := event47533
    frameStart := 47487 },
  { event := event47534
    frameStart := 47487 },
  { event := event47535
    frameStart := 47487 }
]

def eventLeaf2971 : Array AnnotatedEvent := #[
  { event := event47536
    frameStart := 47487 },
  { event := event47537
    frameStart := 47487 },
  { event := event47538
    frameStart := 47487 },
  { event := event47539
    frameStart := 47487 },
  { event := event47540
    frameStart := 47487 },
  { event := event47541
    frameStart := 47487 },
  { event := event47542
    frameStart := 47487 },
  { event := event47543
    frameStart := 47487 },
  { event := event47544
    frameStart := 47487 },
  { event := event47545
    frameStart := 47487 },
  { event := event47546
    frameStart := 47487 },
  { event := event47547
    frameStart := 47487 },
  { event := event47548
    frameStart := 47487 },
  { event := event47549
    frameStart := 47487 },
  { event := event47550
    frameStart := 47487 },
  { event := event47551
    frameStart := 47487 }
]

def eventLeaf2972 : Array AnnotatedEvent := #[
  { event := event47552
    frameStart := 47487 },
  { event := event47553
    frameStart := 47487 },
  { event := event47554
    frameStart := 47487 },
  { event := event47555
    frameStart := 47487 },
  { event := event47556
    frameStart := 47487 },
  { event := event47557
    frameStart := 47487 },
  { event := event47558
    frameStart := 47487 },
  { event := event47559
    frameStart := 47487 },
  { event := event47560
    frameStart := 47487 },
  { event := event47561
    frameStart := 47487 },
  { event := event47562
    frameStart := 47487 },
  { event := event47563
    frameStart := 47487 },
  { event := event47564
    frameStart := 47487 },
  { event := event47565
    frameStart := 47487 },
  { event := event47566
    frameStart := 47487 },
  { event := event47567
    frameStart := 47487 }
]

def eventLeaf2973 : Array AnnotatedEvent := #[
  { event := event47568
    frameStart := 47487 },
  { event := event47569
    frameStart := 47487 },
  { event := event47570
    frameStart := 47487 },
  { event := event47571
    frameStart := 47487 },
  { event := event47572
    frameStart := 47487 },
  { event := event47573
    frameStart := 47487 },
  { event := event47574
    frameStart := 47487 },
  { event := event47575
    frameStart := 47487 },
  { event := event47576
    frameStart := 47487 },
  { event := event47577
    frameStart := 47487 },
  { event := event47578
    frameStart := 47487 },
  { event := event47579
    frameStart := 47487 },
  { event := event47580
    frameStart := 47487 },
  { event := event47581
    frameStart := 47487 },
  { event := event47582
    frameStart := 47487 },
  { event := event47583
    frameStart := 47487 }
]

def eventLeaf2974 : Array AnnotatedEvent := #[
  { event := event47584
    frameStart := 47487 },
  { event := event47585
    frameStart := 47487 },
  { event := event47586
    frameStart := 47487 },
  { event := event47587
    frameStart := 47487 },
  { event := event47588
    frameStart := 47487 },
  { event := event47589
    frameStart := 47487 },
  { event := event47590
    frameStart := 47487 },
  { event := event47591
    frameStart := 0 },
  { event := event47592
    frameStart := 0 },
  { event := event47593
    frameStart := 0 },
  { event := event47594
    frameStart := 0 },
  { event := event47595
    frameStart := 0 },
  { event := event47596
    frameStart := 0 },
  { event := event47597
    frameStart := 0 },
  { event := event47598
    frameStart := 0 },
  { event := event47599
    frameStart := 0 }
]

def eventLeaf2975 : Array AnnotatedEvent := #[
  { event := event47600
    frameStart := 0 },
  { event := event47601
    frameStart := 0 },
  { event := event47602
    frameStart := 0 },
  { event := event47603
    frameStart := 0 },
  { event := event47604
    frameStart := 0 },
  { event := event47605
    frameStart := 0 },
  { event := event47606
    frameStart := 0 },
  { event := event47607
    frameStart := 0 },
  { event := event47608
    frameStart := 0 },
  { event := event47609
    frameStart := 0 },
  { event := event47610
    frameStart := 0 },
  { event := event47611
    frameStart := 0 },
  { event := event47612
    frameStart := 0 },
  { event := event47613
    frameStart := 0 },
  { event := event47614
    frameStart := 0 },
  { event := event47615
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events185
