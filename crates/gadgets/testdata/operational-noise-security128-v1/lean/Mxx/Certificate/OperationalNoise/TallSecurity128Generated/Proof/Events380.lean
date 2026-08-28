import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events380

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event97280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32978⟩⟩) (.finite 3720)

def event97281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32979⟩⟩) 0 ⟨7177⟩ 15500

def event97282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32979⟩⟩) 1 ⟨32978⟩ 97280

def event97283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32979⟩⟩) (.authority (.operator))

def exact97284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (1)⟩]

theorem exact97284RawTermsValid :
    exact97284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32979⟩⟩) exact97284RawTerms .large 97283 .exactZero (none)

def event97285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33514⟩⟩) 0 ⟨32979⟩ 97284

def event97286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33514⟩⟩) (.authority (.operator))

def exact97287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩]

theorem exact97287RawTermsValid :
    exact97287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33514⟩⟩) exact97287RawTerms (.finite 8192) 97286 .exactZero (none)

def event97288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24351⟩⟩) 0 ⟨24350⟩ 4156

def event97289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24351⟩⟩) 1 ⟨9904⟩ 90528

def event97290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24351⟩⟩) (.tensor (.predecessor 0 97288 .coefficient) (.predecessor 1 97289 .coefficient) true false)

def event97291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24351⟩⟩, .operator (⟨4156, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97292RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97292RawTermsValid :
    exact97292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97292 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24351⟩⟩) exact97292RawTerms .large 97290 .exactZero (none)

def event97293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9941⟩⟩) 0 ⟨9903⟩ 90398

def event97294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9941⟩⟩) 1 ⟨7307⟩ 24094

def event97295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9941⟩⟩) (.product (.predecessor 0 97293 .coefficient) (.predecessor 1 97294 .coefficient) (⟨false, false, none, none, none⟩))

def event97296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9941⟩⟩, .operator (⟨90398, 0⟩, ⟨24094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact97297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact97297RawTermsValid :
    exact97297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9941⟩⟩) exact97297RawTerms .large 97295 .exactZero (none)

def event97298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24352⟩⟩) 0 ⟨9941⟩ 97297

def event97299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24352⟩⟩) 1 ⟨24351⟩ 97292

def event97300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24352⟩⟩) (.sum [.predecessor 0 97298 .coefficient, .predecessor 1 97299 .coefficient])

def exact97301RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97301RawTermsValid :
    exact97301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24352⟩⟩) exact97301RawTerms .large 97300 .exactZero (none)

def event97302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24353⟩⟩) 0 ⟨24352⟩ 97301

def event97303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24353⟩⟩) 1 ⟨133⟩ 24086

def event97304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24353⟩⟩) (.sum [.predecessor 0 97302 .coefficient, .predecessor 1 97303 .coefficient])

def event97305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24353⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩) [⟨.result 24086 .coefficient, false, none⟩])

def event97306 : Event := .survivorFold (1) 97305

def exact97307RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97307RawTermsValid :
    exact97307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24353⟩⟩) exact97307RawTerms .large 97304 (.finite 26) (some (97305))

def event97308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31623⟩⟩) 0 ⟨24353⟩ 97307

def event97309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31623⟩⟩) 1 ⟨31620⟩ 4159

def event97310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31623⟩⟩) (.product (.predecessor 0 97308 .coefficient) (.predecessor 1 97309 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31623⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩) [⟨.result 4159 .coefficient, true, some 1⟩])

def event97312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31623⟩⟩) (.product (.result 97307 .summary) (.transfer 97311) (⟨false, false, none, none, none⟩))

def event97313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31623⟩⟩, .operator (⟨97307, 1⟩, ⟨4159, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event97314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31623⟩⟩, .operator (⟨97307, 0⟩, ⟨4159, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact97315RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact97315RawTermsValid :
    exact97315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31623⟩⟩) exact97315RawTerms .large 97310 (.finite 5111808) (some (97312))

def event97316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31624⟩⟩) 0 ⟨31620⟩ 4159

def event97317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31624⟩⟩) 1 ⟨9904⟩ 90528

def event97318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31624⟩⟩) (.tensor (.predecessor 0 97316 .coefficient) (.predecessor 1 97317 .coefficient) true false)

def event97319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31624⟩⟩, .operator (⟨4159, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97320RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97320RawTermsValid :
    exact97320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31624⟩⟩) exact97320RawTerms .large 97318 .exactZero (none)

def event97321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9921⟩⟩) 0 ⟨9903⟩ 90398

def event97322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9921⟩⟩) 1 ⟨7287⟩ 24135

def event97323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9921⟩⟩) (.product (.predecessor 0 97321 .coefficient) (.predecessor 1 97322 .coefficient) (⟨false, false, none, none, none⟩))

def event97324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9921⟩⟩, .operator (⟨90398, 0⟩, ⟨24135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩)

def exact97325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact97325RawTermsValid :
    exact97325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9921⟩⟩) exact97325RawTerms .large 97323 .exactZero (none)

def event97326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31625⟩⟩) 0 ⟨9921⟩ 97325

def event97327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31625⟩⟩) 1 ⟨31624⟩ 97320

def event97328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31625⟩⟩) (.sum [.predecessor 0 97326 .coefficient, .predecessor 1 97327 .coefficient])

def exact97329RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97329RawTermsValid :
    exact97329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31625⟩⟩) exact97329RawTerms .large 97328 .exactZero (none)

def event97330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31626⟩⟩) 0 ⟨31625⟩ 97329

def event97331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31626⟩⟩) 1 ⟨113⟩ 24127

def event97332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31626⟩⟩) (.sum [.predecessor 0 97330 .coefficient, .predecessor 1 97331 .coefficient])

def event97333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31626⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨113⟩⟩]⟩) [⟨.result 24127 .coefficient, false, none⟩])

def event97334 : Event := .survivorFold (1) 97333

def exact97335RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97335RawTermsValid :
    exact97335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97335 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31626⟩⟩) exact97335RawTerms .large 97332 (.finite 26) (some (97333))

def event97336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31627⟩⟩) 0 ⟨31626⟩ 97335

def event97337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31627⟩⟩) 1 ⟨9578⟩ 24124

def event97338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31627⟩⟩) (.product (.predecessor 0 97336 .coefficient) (.predecessor 1 97337 .coefficient) (⟨false, false, none, none, none⟩))

def event97339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31627⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) [⟨.result 24120 .coefficient, false, none⟩])

def event97340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31627⟩⟩) (.product (.result 97335 .summary) (.transfer 97339) (⟨false, false, none, none, none⟩))

def event97341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31627⟩⟩, .operator (⟨97335, 1⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (-1)⟩)

def event97342 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31627⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094)

def event97343 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31627⟩⟩, .relation 97342 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩)

def event97344 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31627⟩⟩, .operator (⟨97335, 0⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact97345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩]

theorem exact97345RawTermsValid :
    exact97345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31627⟩⟩) exact97345RawTerms .large 97338 (.finite 279172874240) (some (97340))

def event97346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31628⟩⟩) 0 ⟨31627⟩ 97345

def event97347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31628⟩⟩) 1 ⟨31623⟩ 97315

def event97348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31628⟩⟩) (.sum [.predecessor 0 97346 .coefficient, .predecessor 1 97347 .coefficient])

def event97349 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31628⟩⟩, .operator (⟨97345, 1⟩, ⟨97315, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def event97350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31628⟩⟩) (.sum [.result 97345 .summary, .result 97315 .summary])

def exact97351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97351RawTermsValid :
    exact97351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31628⟩⟩) exact97351RawTerms .large 97348 (.finite 279177986048) (some (97350))

def event97352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33515⟩⟩) 0 ⟨31628⟩ 97351

def event97353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33515⟩⟩) 1 ⟨33514⟩ 97287

def event97354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33515⟩⟩) (.product (.predecessor 0 97352 .coefficient) (.predecessor 1 97353 .coefficient) (⟨false, false, none, none, none⟩))

def event97355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33515⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩) [⟨.result 97287 .coefficient, false, none⟩])

def event97356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33515⟩⟩) (.product (.result 97351 .summary) (.transfer 97355) (⟨false, false, none, none, none⟩))

def event97357 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33515⟩⟩, .operator (⟨97351, 1⟩, ⟨97287, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (-1)⟩)

def event97358 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33515⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33514⟩⟩) ⟨32979⟩ 97284)

def event97359 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33515⟩⟩, .relation 97358 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (-1)⟩)

def event97360 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33515⟩⟩, .operator (⟨97351, 0⟩, ⟨97287, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩)

def exact97361RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (-1)⟩]

theorem exact97361RawTermsValid :
    exact97361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33515⟩⟩) exact97361RawTerms .large 97354 (.finite 2997650799598260715520) (some (97356))

def event97362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32439⟩⟩) 0 ⟨31622⟩ 4167

def event97363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32439⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact97364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩]

theorem exact97364RawTermsValid :
    exact97364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32439⟩⟩) exact97364RawTerms (.finite 5647228698) 97363 .exactZero (none)

def event97365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32441⟩⟩) 0 ⟨32439⟩ 97364

def event97366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32441⟩⟩) 1 ⟨2370⟩ 4

def event97367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32441⟩⟩) (.scale (.predecessor 0 97365 .coefficient) (.value (.predecessor 1 97366 .coefficient)))

def exact97368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩]

theorem exact97368RawTermsValid :
    exact97368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32441⟩⟩) exact97368RawTerms (.finite 5647228698) 97367 .exactZero (none)

def event97369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32442⟩⟩) 0 ⟨9944⟩ 90620

def event97370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32442⟩⟩) 1 ⟨32441⟩ 97368

def event97371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32442⟩⟩) (.product (.predecessor 0 97369 .coefficient) (.predecessor 1 97370 .coefficient) (⟨false, false, none, none, none⟩))

def event97372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32442⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩) [⟨.result 97364 .coefficient, false, none⟩])

def event97373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32442⟩⟩) (.product (.result 90620 .summary) (.transfer 97372) (⟨false, false, none, none, none⟩))

def event97374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32442⟩⟩, .operator (⟨90620, 0⟩, ⟨97368, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩)

def event97375 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32440⟩⟩)

def event97376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event97377 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event97378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event97379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event97380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event97381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event97382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event97383 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event97384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 97383

def event97385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 97381

def event97386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 97384 .coefficient) (.value (.predecessor 1 97385 .coefficient)))

def event97387 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event97388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 97387

def event97389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 97379

def event97390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 97388 .coefficient, .predecessor 1 97389 .coefficient])

def event97391 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event97392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 97391

def event97393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 97377

def event97394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 97393 .coefficient))

def event97395 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event97396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24350⟩⟩) 0 ⟨9901⟩ 97395

def event97397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24350⟩⟩) (.authority (.programFamilyFact))

def exact97398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩], []⟩, (1)⟩]

theorem exact97398RawTermsValid :
    exact97398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24350⟩⟩) exact97398RawTerms (.finite 6) 97397 .exactZero (none)

def event97399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31620⟩⟩) 0 ⟨9901⟩ 97395

def event97400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31620⟩⟩) (.authority (.programFamilyFact))

def exact97401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩, (1)⟩]

theorem exact97401RawTermsValid :
    exact97401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31620⟩⟩) exact97401RawTerms (.finite 6) 97400 .exactZero (none)

def event97402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31621⟩⟩) 0 ⟨31620⟩ 97401

def event97403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31621⟩⟩) 1 ⟨24350⟩ 97398

def event97404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31621⟩⟩) (.product (.predecessor 0 97402 .coefficient) (.predecessor 1 97403 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31621⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩) [⟨.result 97401 .coefficient, true, some 1⟩, ⟨.result 97398 .coefficient, true, some 1⟩])

def event97406 : Event := .survivorFold (1) 97405

def exact97407RawTerms : List Term := []

theorem exact97407RawTermsValid :
    exact97407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97407 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31621⟩⟩) exact97407RawTerms (.finite 36) 97404 (.finite 36) (some (97405))

def event97408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31622⟩⟩) 0 ⟨31621⟩ 97407

def event97409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31622⟩⟩) (.identity (.predecessor 0 97408 .coefficient))

def event97410 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31622⟩⟩) (.finite 36)

def event97411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32439⟩⟩) 0 ⟨31622⟩ 97410

def event97412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32439⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact97413RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩]

theorem exact97413RawTermsValid :
    exact97413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32439⟩⟩) exact97413RawTerms (.finite 5647228698) 97412 .exactZero (none)

def event97414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact97415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact97415RawTermsValid :
    exact97415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact97415RawTerms .large 97414 .exactZero (none)

def event97416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32440⟩⟩) 0 ⟨35⟩ 97415

def event97417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32440⟩⟩) 1 ⟨32439⟩ 97413

def event97418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32440⟩⟩) (.product (.predecessor 0 97416 .coefficient) (.predecessor 1 97417 .coefficient) (⟨false, false, none, none, none⟩))

def event97419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32440⟩⟩, .operator (⟨97415, 0⟩, ⟨97413, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩)

def exact97420RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩]

theorem exact97420RawTermsValid :
    exact97420RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97420 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32440⟩⟩) exact97420RawTerms .large 97418 .exactZero (none)

def event97421 : Event := .preFoldPolynomial 97420 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩] .exactZero none

def exact97422RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩, (1)⟩]

def event97422 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32440⟩⟩) 97421 exact97422RawTerms .large 97418 .exactZero (none)

def event97423 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33518⟩⟩)

def event97424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event97425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event97426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event97427 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event97428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event97429 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event97430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event97431 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event97432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 97431

def event97433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 97429

def event97434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 97432 .coefficient) (.value (.predecessor 1 97433 .coefficient)))

def event97435 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event97436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 97435

def event97437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 97427

def event97438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 97436 .coefficient, .predecessor 1 97437 .coefficient])

def event97439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event97440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 97439

def event97441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 97425

def event97442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 97441 .coefficient))

def event97443 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event97444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24350⟩⟩) 0 ⟨9901⟩ 97443

def event97445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24350⟩⟩) (.authority (.programFamilyFact))

def exact97446RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩], []⟩, (1)⟩]

theorem exact97446RawTermsValid :
    exact97446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97446 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24350⟩⟩) exact97446RawTerms (.finite 6) 97445 .exactZero (none)

def event97447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31620⟩⟩) 0 ⟨9901⟩ 97443

def event97448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31620⟩⟩) (.authority (.programFamilyFact))

def exact97449RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩, (1)⟩]

theorem exact97449RawTermsValid :
    exact97449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31620⟩⟩) exact97449RawTerms (.finite 6) 97448 .exactZero (none)

def event97450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31621⟩⟩) 0 ⟨31620⟩ 97449

def event97451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31621⟩⟩) 1 ⟨24350⟩ 97446

def event97452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31621⟩⟩) (.product (.predecessor 0 97450 .coefficient) (.predecessor 1 97451 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31621⟩⟩, .operator (⟨97449, 0⟩, ⟨97446, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩, (1)⟩)

def exact97454RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩, (1)⟩]

theorem exact97454RawTermsValid :
    exact97454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31621⟩⟩) exact97454RawTerms (.finite 36) 97452 .exactZero (none)

def event97455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31622⟩⟩) 0 ⟨31621⟩ 97454

def event97456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31622⟩⟩) (.identity (.predecessor 0 97455 .coefficient))

def event97457 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31622⟩⟩) (.finite 36)

def event97458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32978⟩⟩) 0 ⟨31622⟩ 97457

def event97459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32978⟩⟩) (.authority (.programFamilyFact))

def event97460 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32978⟩⟩) (.finite 3720)

def event97461 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event97462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32979⟩⟩) 0 ⟨7177⟩ 97461

def event97463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32979⟩⟩) 1 ⟨32978⟩ 97460

def event97464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32979⟩⟩) (.authority (.operator))

def exact97465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (1)⟩]

theorem exact97465RawTermsValid :
    exact97465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97465 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32979⟩⟩) exact97465RawTerms .large 97464 .exactZero (none)

def event97466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33514⟩⟩) 0 ⟨32979⟩ 97465

def event97467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33514⟩⟩) (.authority (.operator))

def exact97468RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩]

theorem exact97468RawTermsValid :
    exact97468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33514⟩⟩) exact97468RawTerms (.finite 8192) 97467 .exactZero (none)

def event97469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event97470 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event97471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33246⟩⟩) 0 ⟨31622⟩ 97457

def event97472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33246⟩⟩) 1 ⟨136⟩ 97470

def event97473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33246⟩⟩) (.sum [.predecessor 0 97471 .coefficient, .predecessor 1 97472 .coefficient])

def event97474 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33246⟩⟩) (.finite 36)

def event97475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33247⟩⟩) 0 ⟨33246⟩ 97474

def event97476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33247⟩⟩) (.identity (.predecessor 0 97475 .coefficient))

def exact97477RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], []⟩, (1)⟩]

theorem exact97477RawTermsValid :
    exact97477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97477 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33247⟩⟩) exact97477RawTerms (.finite 36) 97476 .exactZero (none)

def event97478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact97479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97479RawTermsValid :
    exact97479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact97479RawTerms .large 97478 .exactZero (none)

def event97480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33248⟩⟩) 0 ⟨6908⟩ 97479

def event97481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33248⟩⟩) 1 ⟨33247⟩ 97477

def event97482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33248⟩⟩) (.product (.predecessor 0 97480 .coefficient) (.predecessor 1 97481 .coefficient) (⟨false, false, none, none, none⟩))

def event97483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33248⟩⟩, .operator (⟨97479, 0⟩, ⟨97477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97484RawTermsValid :
    exact97484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33248⟩⟩) exact97484RawTerms .large 97482 .exactZero (none)

def event97485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event97486 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event97487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 97461

def event97488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact97489RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact97489RawTermsValid :
    exact97489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97489 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact97489RawTerms .large 97488 .exactZero (none)

def event97490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7307⟩⟩) 0 ⟨7178⟩ 97489

def event97491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7307⟩⟩) (.identity (.predecessor 0 97490 .coefficient))

def exact97492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact97492RawTermsValid :
    exact97492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7307⟩⟩) exact97492RawTerms .large 97491 .exactZero (none)

def event97493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9577⟩⟩) 0 ⟨7307⟩ 97492

def event97494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9577⟩⟩) (.authority (.operator))

def exact97495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact97495RawTermsValid :
    exact97495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9577⟩⟩) exact97495RawTerms (.finite 8192) 97494 .exactZero (none)

def event97496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 0 ⟨9577⟩ 97495

def event97497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 1 ⟨2370⟩ 97486

def event97498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9578⟩⟩) (.scale (.predecessor 0 97496 .coefficient) (.value (.predecessor 1 97497 .coefficient)))

def exact97499RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact97499RawTermsValid :
    exact97499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9578⟩⟩) exact97499RawTerms (.finite 8192) 97498 .exactZero (none)

def event97500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7287⟩⟩) 0 ⟨7178⟩ 97489

def event97501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7287⟩⟩) (.identity (.predecessor 0 97500 .coefficient))

def exact97502RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact97502RawTermsValid :
    exact97502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97502 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7287⟩⟩) exact97502RawTerms .large 97501 .exactZero (none)

def event97503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 0 ⟨7287⟩ 97502

def event97504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 1 ⟨9578⟩ 97499

def event97505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9579⟩⟩) (.product (.predecessor 0 97503 .coefficient) (.predecessor 1 97504 .coefficient) (⟨false, false, none, none, none⟩))

def event97506 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9579⟩⟩, .operator (⟨97502, 0⟩, ⟨97499, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact97507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact97507RawTermsValid :
    exact97507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9579⟩⟩) exact97507RawTerms .large 97505 .exactZero (none)

def event97508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33249⟩⟩) 0 ⟨9579⟩ 97507

def event97509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33249⟩⟩) 1 ⟨33248⟩ 97484

def event97510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33249⟩⟩) (.sum [.predecessor 0 97508 .coefficient, .predecessor 1 97509 .coefficient])

def exact97511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97511RawTermsValid :
    exact97511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33249⟩⟩) exact97511RawTerms .large 97510 .exactZero (none)

def event97512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33517⟩⟩) 0 ⟨33249⟩ 97511

def event97513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33517⟩⟩) 1 ⟨33514⟩ 97468

def event97514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33517⟩⟩) (.product (.predecessor 0 97512 .coefficient) (.predecessor 1 97513 .coefficient) (⟨false, false, none, none, none⟩))

def event97515 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33517⟩⟩, .operator (⟨97511, 0⟩, ⟨97468, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩)

def event97516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33517⟩⟩, .operator (⟨97511, 1⟩, ⟨97468, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (-1)⟩)

def event97517 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33517⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33514⟩⟩) ⟨32979⟩ 97465)

def event97518 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33517⟩⟩, .relation 97517 0, ⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (-1)⟩)

def exact97519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩, (-1)⟩]

theorem exact97519RawTermsValid :
    exact97519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33517⟩⟩) exact97519RawTerms .large 97514 .exactZero (none)

def event97520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31868⟩⟩) 0 ⟨31622⟩ 97457

def event97521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31868⟩⟩) (.authority (.programFamilyFact))

def exact97522RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], []⟩, (1)⟩]

theorem exact97522RawTermsValid :
    exact97522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31868⟩⟩) exact97522RawTerms (.finite 6) 97521 .exactZero (none)

def event97523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31870⟩⟩) 0 ⟨6908⟩ 97479

def event97524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31870⟩⟩) 1 ⟨31868⟩ 97522

def event97525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31870⟩⟩) (.product (.predecessor 0 97523 .coefficient) (.predecessor 1 97524 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31870⟩⟩, .operator (⟨97479, 0⟩, ⟨97522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97527RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97527RawTermsValid :
    exact97527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31870⟩⟩) exact97527RawTerms .large 97525 .exactZero (none)

def event97528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 97461

def event97529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact97530RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact97530RawTermsValid :
    exact97530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact97530RawTerms .large 97529 .exactZero (none)

def event97531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31871⟩⟩) 0 ⟨7182⟩ 97530

def event97532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31871⟩⟩) 1 ⟨31870⟩ 97527

def event97533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31871⟩⟩) (.sum [.predecessor 0 97531 .coefficient, .predecessor 1 97532 .coefficient])

def exact97534RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97534RawTermsValid :
    exact97534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97534 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31871⟩⟩) exact97534RawTerms .large 97533 .exactZero (none)

def event97535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33518⟩⟩) 0 ⟨31871⟩ 97534

def eventLeaf6080 : Array AnnotatedEvent := #[
  { event := event97280
    frameStart := 0 },
  { event := event97281
    frameStart := 0 },
  { event := event97282
    frameStart := 0 },
  { event := event97283
    frameStart := 0 },
  { event := event97284
    frameStart := 0 },
  { event := event97285
    frameStart := 0 },
  { event := event97286
    frameStart := 0 },
  { event := event97287
    frameStart := 0 },
  { event := event97288
    frameStart := 0 },
  { event := event97289
    frameStart := 0 },
  { event := event97290
    frameStart := 0 },
  { event := event97291
    frameStart := 0 },
  { event := event97292
    frameStart := 0 },
  { event := event97293
    frameStart := 0 },
  { event := event97294
    frameStart := 0 },
  { event := event97295
    frameStart := 0 }
]

def eventLeaf6081 : Array AnnotatedEvent := #[
  { event := event97296
    frameStart := 0 },
  { event := event97297
    frameStart := 0 },
  { event := event97298
    frameStart := 0 },
  { event := event97299
    frameStart := 0 },
  { event := event97300
    frameStart := 0 },
  { event := event97301
    frameStart := 0 },
  { event := event97302
    frameStart := 0 },
  { event := event97303
    frameStart := 0 },
  { event := event97304
    frameStart := 0 },
  { event := event97305
    frameStart := 0 },
  { event := event97306
    frameStart := 0 },
  { event := event97307
    frameStart := 0 },
  { event := event97308
    frameStart := 0 },
  { event := event97309
    frameStart := 0 },
  { event := event97310
    frameStart := 0 },
  { event := event97311
    frameStart := 0 }
]

def eventLeaf6082 : Array AnnotatedEvent := #[
  { event := event97312
    frameStart := 0 },
  { event := event97313
    frameStart := 0 },
  { event := event97314
    frameStart := 0 },
  { event := event97315
    frameStart := 0 },
  { event := event97316
    frameStart := 0 },
  { event := event97317
    frameStart := 0 },
  { event := event97318
    frameStart := 0 },
  { event := event97319
    frameStart := 0 },
  { event := event97320
    frameStart := 0 },
  { event := event97321
    frameStart := 0 },
  { event := event97322
    frameStart := 0 },
  { event := event97323
    frameStart := 0 },
  { event := event97324
    frameStart := 0 },
  { event := event97325
    frameStart := 0 },
  { event := event97326
    frameStart := 0 },
  { event := event97327
    frameStart := 0 }
]

def eventLeaf6083 : Array AnnotatedEvent := #[
  { event := event97328
    frameStart := 0 },
  { event := event97329
    frameStart := 0 },
  { event := event97330
    frameStart := 0 },
  { event := event97331
    frameStart := 0 },
  { event := event97332
    frameStart := 0 },
  { event := event97333
    frameStart := 0 },
  { event := event97334
    frameStart := 0 },
  { event := event97335
    frameStart := 0 },
  { event := event97336
    frameStart := 0 },
  { event := event97337
    frameStart := 0 },
  { event := event97338
    frameStart := 0 },
  { event := event97339
    frameStart := 0 },
  { event := event97340
    frameStart := 0 },
  { event := event97341
    frameStart := 0 },
  { event := event97342
    frameStart := 0 },
  { event := event97343
    frameStart := 0 }
]

def eventLeaf6084 : Array AnnotatedEvent := #[
  { event := event97344
    frameStart := 0 },
  { event := event97345
    frameStart := 0 },
  { event := event97346
    frameStart := 0 },
  { event := event97347
    frameStart := 0 },
  { event := event97348
    frameStart := 0 },
  { event := event97349
    frameStart := 0 },
  { event := event97350
    frameStart := 0 },
  { event := event97351
    frameStart := 0 },
  { event := event97352
    frameStart := 0 },
  { event := event97353
    frameStart := 0 },
  { event := event97354
    frameStart := 0 },
  { event := event97355
    frameStart := 0 },
  { event := event97356
    frameStart := 0 },
  { event := event97357
    frameStart := 0 },
  { event := event97358
    frameStart := 0 },
  { event := event97359
    frameStart := 0 }
]

def eventLeaf6085 : Array AnnotatedEvent := #[
  { event := event97360
    frameStart := 0 },
  { event := event97361
    frameStart := 0 },
  { event := event97362
    frameStart := 0 },
  { event := event97363
    frameStart := 0 },
  { event := event97364
    frameStart := 0 },
  { event := event97365
    frameStart := 0 },
  { event := event97366
    frameStart := 0 },
  { event := event97367
    frameStart := 0 },
  { event := event97368
    frameStart := 0 },
  { event := event97369
    frameStart := 0 },
  { event := event97370
    frameStart := 0 },
  { event := event97371
    frameStart := 0 },
  { event := event97372
    frameStart := 0 },
  { event := event97373
    frameStart := 0 },
  { event := event97374
    frameStart := 0 },
  { event := event97375
    frameStart := 97375 }
]

def eventLeaf6086 : Array AnnotatedEvent := #[
  { event := event97376
    frameStart := 97375 },
  { event := event97377
    frameStart := 97375 },
  { event := event97378
    frameStart := 97375 },
  { event := event97379
    frameStart := 97375 },
  { event := event97380
    frameStart := 97375 },
  { event := event97381
    frameStart := 97375 },
  { event := event97382
    frameStart := 97375 },
  { event := event97383
    frameStart := 97375 },
  { event := event97384
    frameStart := 97375 },
  { event := event97385
    frameStart := 97375 },
  { event := event97386
    frameStart := 97375 },
  { event := event97387
    frameStart := 97375 },
  { event := event97388
    frameStart := 97375 },
  { event := event97389
    frameStart := 97375 },
  { event := event97390
    frameStart := 97375 },
  { event := event97391
    frameStart := 97375 }
]

def eventLeaf6087 : Array AnnotatedEvent := #[
  { event := event97392
    frameStart := 97375 },
  { event := event97393
    frameStart := 97375 },
  { event := event97394
    frameStart := 97375 },
  { event := event97395
    frameStart := 97375 },
  { event := event97396
    frameStart := 97375 },
  { event := event97397
    frameStart := 97375 },
  { event := event97398
    frameStart := 97375 },
  { event := event97399
    frameStart := 97375 },
  { event := event97400
    frameStart := 97375 },
  { event := event97401
    frameStart := 97375 },
  { event := event97402
    frameStart := 97375 },
  { event := event97403
    frameStart := 97375 },
  { event := event97404
    frameStart := 97375 },
  { event := event97405
    frameStart := 97375 },
  { event := event97406
    frameStart := 97375 },
  { event := event97407
    frameStart := 97375 }
]

def eventLeaf6088 : Array AnnotatedEvent := #[
  { event := event97408
    frameStart := 97375 },
  { event := event97409
    frameStart := 97375 },
  { event := event97410
    frameStart := 97375 },
  { event := event97411
    frameStart := 97375 },
  { event := event97412
    frameStart := 97375 },
  { event := event97413
    frameStart := 97375 },
  { event := event97414
    frameStart := 97375 },
  { event := event97415
    frameStart := 97375 },
  { event := event97416
    frameStart := 97375 },
  { event := event97417
    frameStart := 97375 },
  { event := event97418
    frameStart := 97375 },
  { event := event97419
    frameStart := 97375 },
  { event := event97420
    frameStart := 97375 },
  { event := event97421
    frameStart := 97375 },
  { event := event97422
    frameStart := 97375 },
  { event := event97423
    frameStart := 97423 }
]

def eventLeaf6089 : Array AnnotatedEvent := #[
  { event := event97424
    frameStart := 97423 },
  { event := event97425
    frameStart := 97423 },
  { event := event97426
    frameStart := 97423 },
  { event := event97427
    frameStart := 97423 },
  { event := event97428
    frameStart := 97423 },
  { event := event97429
    frameStart := 97423 },
  { event := event97430
    frameStart := 97423 },
  { event := event97431
    frameStart := 97423 },
  { event := event97432
    frameStart := 97423 },
  { event := event97433
    frameStart := 97423 },
  { event := event97434
    frameStart := 97423 },
  { event := event97435
    frameStart := 97423 },
  { event := event97436
    frameStart := 97423 },
  { event := event97437
    frameStart := 97423 },
  { event := event97438
    frameStart := 97423 },
  { event := event97439
    frameStart := 97423 }
]

def eventLeaf6090 : Array AnnotatedEvent := #[
  { event := event97440
    frameStart := 97423 },
  { event := event97441
    frameStart := 97423 },
  { event := event97442
    frameStart := 97423 },
  { event := event97443
    frameStart := 97423 },
  { event := event97444
    frameStart := 97423 },
  { event := event97445
    frameStart := 97423 },
  { event := event97446
    frameStart := 97423 },
  { event := event97447
    frameStart := 97423 },
  { event := event97448
    frameStart := 97423 },
  { event := event97449
    frameStart := 97423 },
  { event := event97450
    frameStart := 97423 },
  { event := event97451
    frameStart := 97423 },
  { event := event97452
    frameStart := 97423 },
  { event := event97453
    frameStart := 97423 },
  { event := event97454
    frameStart := 97423 },
  { event := event97455
    frameStart := 97423 }
]

def eventLeaf6091 : Array AnnotatedEvent := #[
  { event := event97456
    frameStart := 97423 },
  { event := event97457
    frameStart := 97423 },
  { event := event97458
    frameStart := 97423 },
  { event := event97459
    frameStart := 97423 },
  { event := event97460
    frameStart := 97423 },
  { event := event97461
    frameStart := 97423 },
  { event := event97462
    frameStart := 97423 },
  { event := event97463
    frameStart := 97423 },
  { event := event97464
    frameStart := 97423 },
  { event := event97465
    frameStart := 97423 },
  { event := event97466
    frameStart := 97423 },
  { event := event97467
    frameStart := 97423 },
  { event := event97468
    frameStart := 97423 },
  { event := event97469
    frameStart := 97423 },
  { event := event97470
    frameStart := 97423 },
  { event := event97471
    frameStart := 97423 }
]

def eventLeaf6092 : Array AnnotatedEvent := #[
  { event := event97472
    frameStart := 97423 },
  { event := event97473
    frameStart := 97423 },
  { event := event97474
    frameStart := 97423 },
  { event := event97475
    frameStart := 97423 },
  { event := event97476
    frameStart := 97423 },
  { event := event97477
    frameStart := 97423 },
  { event := event97478
    frameStart := 97423 },
  { event := event97479
    frameStart := 97423 },
  { event := event97480
    frameStart := 97423 },
  { event := event97481
    frameStart := 97423 },
  { event := event97482
    frameStart := 97423 },
  { event := event97483
    frameStart := 97423 },
  { event := event97484
    frameStart := 97423 },
  { event := event97485
    frameStart := 97423 },
  { event := event97486
    frameStart := 97423 },
  { event := event97487
    frameStart := 97423 }
]

def eventLeaf6093 : Array AnnotatedEvent := #[
  { event := event97488
    frameStart := 97423 },
  { event := event97489
    frameStart := 97423 },
  { event := event97490
    frameStart := 97423 },
  { event := event97491
    frameStart := 97423 },
  { event := event97492
    frameStart := 97423 },
  { event := event97493
    frameStart := 97423 },
  { event := event97494
    frameStart := 97423 },
  { event := event97495
    frameStart := 97423 },
  { event := event97496
    frameStart := 97423 },
  { event := event97497
    frameStart := 97423 },
  { event := event97498
    frameStart := 97423 },
  { event := event97499
    frameStart := 97423 },
  { event := event97500
    frameStart := 97423 },
  { event := event97501
    frameStart := 97423 },
  { event := event97502
    frameStart := 97423 },
  { event := event97503
    frameStart := 97423 }
]

def eventLeaf6094 : Array AnnotatedEvent := #[
  { event := event97504
    frameStart := 97423 },
  { event := event97505
    frameStart := 97423 },
  { event := event97506
    frameStart := 97423 },
  { event := event97507
    frameStart := 97423 },
  { event := event97508
    frameStart := 97423 },
  { event := event97509
    frameStart := 97423 },
  { event := event97510
    frameStart := 97423 },
  { event := event97511
    frameStart := 97423 },
  { event := event97512
    frameStart := 97423 },
  { event := event97513
    frameStart := 97423 },
  { event := event97514
    frameStart := 97423 },
  { event := event97515
    frameStart := 97423 },
  { event := event97516
    frameStart := 97423 },
  { event := event97517
    frameStart := 97423 },
  { event := event97518
    frameStart := 97423 },
  { event := event97519
    frameStart := 97423 }
]

def eventLeaf6095 : Array AnnotatedEvent := #[
  { event := event97520
    frameStart := 97423 },
  { event := event97521
    frameStart := 97423 },
  { event := event97522
    frameStart := 97423 },
  { event := event97523
    frameStart := 97423 },
  { event := event97524
    frameStart := 97423 },
  { event := event97525
    frameStart := 97423 },
  { event := event97526
    frameStart := 97423 },
  { event := event97527
    frameStart := 97423 },
  { event := event97528
    frameStart := 97423 },
  { event := event97529
    frameStart := 97423 },
  { event := event97530
    frameStart := 97423 },
  { event := event97531
    frameStart := 97423 },
  { event := event97532
    frameStart := 97423 },
  { event := event97533
    frameStart := 97423 },
  { event := event97534
    frameStart := 97423 },
  { event := event97535
    frameStart := 97423 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events380
