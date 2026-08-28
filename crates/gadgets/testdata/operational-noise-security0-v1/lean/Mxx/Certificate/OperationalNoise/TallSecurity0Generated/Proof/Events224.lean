import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events224

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event57344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event57345 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event57346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15662⟩⟩) 0 ⟨15588⟩ 57332

def event57347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15662⟩⟩) 1 ⟨110⟩ 57345

def event57348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15662⟩⟩) (.sum [.predecessor 0 57346 .coefficient, .predecessor 1 57347 .coefficient])

def event57349 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15662⟩⟩) (.finite 10)

def event57350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15663⟩⟩) 0 ⟨15662⟩ 57349

def event57351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15663⟩⟩) (.identity (.predecessor 0 57350 .coefficient))

def exact57352RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], []⟩, (1)⟩]

theorem exact57352RawTermsValid :
    exact57352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57352 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15663⟩⟩) exact57352RawTerms (.finite 10) 57351 .exactZero (none)

def event57353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact57354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact57354RawTermsValid :
    exact57354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57354 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact57354RawTerms .large 57353 .exactZero (none)

def event57355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15664⟩⟩) 0 ⟨6544⟩ 57354

def event57356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15664⟩⟩) 1 ⟨15663⟩ 57352

def event57357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15664⟩⟩) (.product (.predecessor 0 57355 .coefficient) (.predecessor 1 57356 .coefficient) (⟨false, false, none, none, none⟩))

def event57358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15664⟩⟩, .operator (⟨57354, 0⟩, ⟨57352, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact57359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact57359RawTermsValid :
    exact57359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15664⟩⟩) exact57359RawTerms .large 57357 .exactZero (none)

def event57360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 57336

def event57361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact57362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact57362RawTermsValid :
    exact57362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57362 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact57362RawTerms .large 57361 .exactZero (none)

def event57363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15665⟩⟩) 0 ⟨6694⟩ 57362

def event57364 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15665⟩⟩) 1 ⟨15664⟩ 57359

def event57365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15665⟩⟩) (.sum [.predecessor 0 57363 .coefficient, .predecessor 1 57364 .coefficient])

def exact57366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57366RawTermsValid :
    exact57366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57366 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15665⟩⟩) exact57366RawTerms .large 57365 .exactZero (none)

def event57367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27229⟩⟩) 0 ⟨15665⟩ 57366

def event57368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27229⟩⟩) 1 ⟨27228⟩ 57343

def event57369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27229⟩⟩) (.product (.predecessor 0 57367 .coefficient) (.predecessor 1 57368 .coefficient) (⟨false, false, none, none, none⟩))

def event57370 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27229⟩⟩, .operator (⟨57366, 0⟩, ⟨57343, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (1)⟩)

def event57371 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27229⟩⟩, .operator (⟨57366, 1⟩, ⟨57343, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩)

def event57372 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27229⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27228⟩⟩) ⟨23976⟩ 57340)

def event57373 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27229⟩⟩, .relation 57372 0, ⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (-1)⟩)

def exact57374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (-1)⟩]

theorem exact57374RawTermsValid :
    exact57374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27229⟩⟩) exact57374RawTerms .large 57369 .exactZero (none)

def event57375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15632⟩⟩) 0 ⟨15588⟩ 57332

def event57376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15632⟩⟩) (.authority (.programFamilyFact))

def exact57377RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩]

theorem exact57377RawTermsValid :
    exact57377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15632⟩⟩) exact57377RawTerms (.finite 58) 57376 .exactZero (none)

def event57378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15633⟩⟩) 0 ⟨6544⟩ 57354

def event57379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15633⟩⟩) 1 ⟨15632⟩ 57377

def event57380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15633⟩⟩) (.product (.predecessor 0 57378 .coefficient) (.predecessor 1 57379 .coefficient) (⟨false, true, none, none, some 1⟩))

def event57381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15633⟩⟩, .operator (⟨57354, 0⟩, ⟨57377, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact57382RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact57382RawTermsValid :
    exact57382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57382 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15633⟩⟩) exact57382RawTerms .large 57380 .exactZero (none)

def event57383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6717⟩⟩) 0 ⟨6689⟩ 57336

def event57384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6717⟩⟩) (.authority (.operator))

def exact57385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩]

theorem exact57385RawTermsValid :
    exact57385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6717⟩⟩) exact57385RawTerms .large 57384 .exactZero (none)

def event57386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15634⟩⟩) 0 ⟨6717⟩ 57385

def event57387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15634⟩⟩) 1 ⟨15633⟩ 57382

def event57388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15634⟩⟩) (.sum [.predecessor 0 57386 .coefficient, .predecessor 1 57387 .coefficient])

def exact57389RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57389RawTermsValid :
    exact57389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57389 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15634⟩⟩) exact57389RawTerms .large 57388 .exactZero (none)

def event57390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27233⟩⟩) 0 ⟨15634⟩ 57389

def event57391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27233⟩⟩) 1 ⟨27229⟩ 57374

def event57392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27233⟩⟩) (.sum [.predecessor 0 57390 .coefficient, .predecessor 1 57391 .coefficient])

def exact57393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57393RawTermsValid :
    exact57393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57393 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27233⟩⟩) exact57393RawTerms .large 57392 .exactZero (none)

def event57394 : Event := .preFoldPolynomial 57393 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact57395RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event57395 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27233⟩⟩) 57394 exact57395RawTerms .large 57392 .exactZero (none)

def event57396 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15588⟩⟩) ⟨⟨130⟩, ⟨37⟩, ⟨109⟩⟩ ⟨57238, 57396⟩

def event57397 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20975⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20972⟩⟩]⟩) (1) 0 2 (.universal 57396 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20972⟩⟩]⟩) (none) 57395)

def event57398 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20975⟩⟩, .relation 57397 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩)

def event57399 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20975⟩⟩, .relation 57397 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩)

def event57400 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20975⟩⟩, .relation 57397 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (1)⟩)

def event57401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20975⟩⟩, .relation 57397 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact57402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57402RawTermsValid :
    exact57402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20975⟩⟩) exact57402RawTerms .large 57234 (.finite 1811303510016) (some (57236))

def event57403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27231⟩⟩) 0 ⟨20975⟩ 57402

def event57404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27231⟩⟩) 1 ⟨27230⟩ 57224

def event57405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27231⟩⟩) (.sum [.predecessor 0 57403 .coefficient, .predecessor 1 57404 .coefficient])

def event57406 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27231⟩⟩, .operator (⟨57402, 0⟩, ⟨57224, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27228⟩⟩]⟩, (1)⟩)

def event57407 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27231⟩⟩, .operator (⟨57402, 2⟩, ⟨57224, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15587⟩⟩], [⟨.program ⟨214⟩, ⟨23976⟩⟩]⟩, (-1)⟩)

def event57408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27231⟩⟩) (.sum [.result 57402 .summary, .result 57224 .summary])

def exact57409RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57409RawTermsValid :
    exact57409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27231⟩⟩) exact57409RawTerms .large 57405 (.finite 1291978824159503986688) (some (57408))

def event57410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23911⟩⟩) 0 ⟨15427⟩ 2677

def event57411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23911⟩⟩) (.authority (.programFamilyFact))

def event57412 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23911⟩⟩) (.finite 3720)

def event57413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23913⟩⟩) 0 ⟨6689⟩ 5477

def event57414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23913⟩⟩) 1 ⟨23911⟩ 57412

def event57415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23913⟩⟩) (.authority (.operator))

def exact57416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23913⟩⟩]⟩, (1)⟩]

theorem exact57416RawTermsValid :
    exact57416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23913⟩⟩) exact57416RawTerms .large 57415 .exactZero (none)

def event57417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27011⟩⟩) 0 ⟨23913⟩ 57416

def event57418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27011⟩⟩) (.authority (.operator))

def exact57419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27011⟩⟩]⟩, (1)⟩]

theorem exact57419RawTermsValid :
    exact57419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27011⟩⟩) exact57419RawTerms (.finite 8192) 57418 .exactZero (none)

def event57420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23165⟩⟩) 0 ⟨12174⟩ 2671

def event57421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23165⟩⟩) (.authority (.programFamilyFact))

def event57422 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23165⟩⟩) (.finite 3720)

def event57423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23166⟩⟩) 0 ⟨6689⟩ 5477

def event57424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23166⟩⟩) 1 ⟨23165⟩ 57422

def event57425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23166⟩⟩) (.authority (.operator))

def exact57426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23166⟩⟩]⟩, (1)⟩]

theorem exact57426RawTermsValid :
    exact57426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23166⟩⟩) exact57426RawTerms .large 57425 .exactZero (none)

def event57427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25301⟩⟩) 0 ⟨23166⟩ 57426

def event57428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25301⟩⟩) (.authority (.operator))

def exact57429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩, (1)⟩]

theorem exact57429RawTermsValid :
    exact57429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25301⟩⟩) exact57429RawTerms (.finite 8192) 57428 .exactZero (none)

def event57430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11138⟩⟩) 0 ⟨11137⟩ 2660

def event57431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11138⟩⟩) 1 ⟨6568⟩ 50670

def event57432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11138⟩⟩) (.tensor (.predecessor 0 57430 .coefficient) (.predecessor 1 57431 .coefficient) true false)

def event57433 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11138⟩⟩, .operator (⟨2660, 0⟩, ⟨50670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact57434RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact57434RawTermsValid :
    exact57434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57434 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11138⟩⟩) exact57434RawTerms .large 57432 .exactZero (none)

def event57435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7269⟩⟩) 0 ⟨5545⟩ 50540

def event57436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7269⟩⟩) 1 ⟨6775⟩ 13486

def event57437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7269⟩⟩) (.product (.predecessor 0 57435 .coefficient) (.predecessor 1 57436 .coefficient) (⟨false, false, none, none, none⟩))

def event57438 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7269⟩⟩, .operator (⟨50540, 0⟩, ⟨13486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact57439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact57439RawTermsValid :
    exact57439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7269⟩⟩) exact57439RawTerms .large 57437 .exactZero (none)

def event57440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11139⟩⟩) 0 ⟨7269⟩ 57439

def event57441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11139⟩⟩) 1 ⟨11138⟩ 57434

def event57442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11139⟩⟩) (.sum [.predecessor 0 57440 .coefficient, .predecessor 1 57441 .coefficient])

def exact57443RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57443RawTermsValid :
    exact57443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11139⟩⟩) exact57443RawTerms .large 57442 .exactZero (none)

def event57444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11140⟩⟩) 0 ⟨11139⟩ 57443

def event57445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11140⟩⟩) 1 ⟨89⟩ 13478

def event57446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11140⟩⟩) (.sum [.predecessor 0 57444 .coefficient, .predecessor 1 57445 .coefficient])

def event57447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11140⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) [⟨.result 13478 .coefficient, false, none⟩])

def event57448 : Event := .survivorFold (1) 57447

def exact57449RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57449RawTermsValid :
    exact57449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57449 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11140⟩⟩) exact57449RawTerms .large 57446 (.finite 26) (some (57447))

def event57450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12175⟩⟩) 0 ⟨11140⟩ 57449

def event57451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12175⟩⟩) 1 ⟨12172⟩ 2663

def event57452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12175⟩⟩) (.product (.predecessor 0 57450 .coefficient) (.predecessor 1 57451 .coefficient) (⟨false, true, none, none, some 1⟩))

def event57453 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12175⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩) [⟨.result 2663 .coefficient, true, some 1⟩])

def event57454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12175⟩⟩) (.product (.result 57449 .summary) (.transfer 57453) (⟨false, false, none, none, none⟩))

def event57455 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12175⟩⟩, .operator (⟨57449, 1⟩, ⟨2663, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event57456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12175⟩⟩, .operator (⟨57449, 0⟩, ⟨2663, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact57457RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact57457RawTermsValid :
    exact57457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12175⟩⟩) exact57457RawTerms .large 57452 (.finite 4992) (some (57454))

def event57458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12176⟩⟩) 0 ⟨12172⟩ 2663

def event57459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12176⟩⟩) 1 ⟨6568⟩ 50670

def event57460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12176⟩⟩) (.tensor (.predecessor 0 57458 .coefficient) (.predecessor 1 57459 .coefficient) true false)

def event57461 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12176⟩⟩, .operator (⟨2663, 0⟩, ⟨50670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact57462RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact57462RawTermsValid :
    exact57462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12176⟩⟩) exact57462RawTerms .large 57460 .exactZero (none)

def event57463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7286⟩⟩) 0 ⟨5545⟩ 50540

def event57464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7286⟩⟩) 1 ⟨6792⟩ 13527

def event57465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7286⟩⟩) (.product (.predecessor 0 57463 .coefficient) (.predecessor 1 57464 .coefficient) (⟨false, false, none, none, none⟩))

def event57466 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7286⟩⟩, .operator (⟨50540, 0⟩, ⟨13527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩)

def exact57467RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact57467RawTermsValid :
    exact57467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57467 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7286⟩⟩) exact57467RawTerms .large 57465 .exactZero (none)

def event57468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12177⟩⟩) 0 ⟨7286⟩ 57467

def event57469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12177⟩⟩) 1 ⟨12176⟩ 57462

def event57470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12177⟩⟩) (.sum [.predecessor 0 57468 .coefficient, .predecessor 1 57469 .coefficient])

def exact57471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57471RawTermsValid :
    exact57471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12177⟩⟩) exact57471RawTerms .large 57470 .exactZero (none)

def event57472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12178⟩⟩) 0 ⟨12177⟩ 57471

def event57473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12178⟩⟩) 1 ⟨106⟩ 13519

def event57474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12178⟩⟩) (.sum [.predecessor 0 57472 .coefficient, .predecessor 1 57473 .coefficient])

def event57475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12178⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) [⟨.result 13519 .coefficient, false, none⟩])

def event57476 : Event := .survivorFold (1) 57475

def exact57477RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57477RawTermsValid :
    exact57477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57477 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12178⟩⟩) exact57477RawTerms .large 57474 (.finite 26) (some (57475))

def event57478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12179⟩⟩) 0 ⟨12178⟩ 57477

def event57479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12179⟩⟩) 1 ⟨7841⟩ 13516

def event57480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12179⟩⟩) (.product (.predecessor 0 57478 .coefficient) (.predecessor 1 57479 .coefficient) (⟨false, false, none, none, none⟩))

def event57481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12179⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) [⟨.result 13512 .coefficient, false, none⟩])

def event57482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12179⟩⟩) (.product (.result 57477 .summary) (.transfer 57481) (⟨false, false, none, none, none⟩))

def event57483 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12179⟩⟩, .operator (⟨57477, 1⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (-1)⟩)

def event57484 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨12179⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7840⟩⟩) ⟨6775⟩ 13486)

def event57485 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12179⟩⟩, .relation 57484 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩)

def event57486 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12179⟩⟩, .operator (⟨57477, 0⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact57487RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩]

theorem exact57487RawTermsValid :
    exact57487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57487 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12179⟩⟩) exact57487RawTerms .large 57480 (.finite 95420416) (some (57482))

def event57488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12180⟩⟩) 0 ⟨12179⟩ 57487

def event57489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12180⟩⟩) 1 ⟨12175⟩ 57457

def event57490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12180⟩⟩) (.sum [.predecessor 0 57488 .coefficient, .predecessor 1 57489 .coefficient])

def event57491 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12180⟩⟩, .operator (⟨57487, 1⟩, ⟨57457, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def event57492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12180⟩⟩) (.sum [.result 57487 .summary, .result 57457 .summary])

def exact57493RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact57493RawTermsValid :
    exact57493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57493 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12180⟩⟩) exact57493RawTerms .large 57490 (.finite 95425408) (some (57492))

def event57494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25302⟩⟩) 0 ⟨12180⟩ 57493

def event57495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25302⟩⟩) 1 ⟨25301⟩ 57429

def event57496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25302⟩⟩) (.product (.predecessor 0 57494 .coefficient) (.predecessor 1 57495 .coefficient) (⟨false, false, none, none, none⟩))

def event57497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25302⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) [⟨.result 57429 .coefficient, false, none⟩])

def event57498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25302⟩⟩) (.product (.result 57493 .summary) (.transfer 57497) (⟨false, false, none, none, none⟩))

def event57499 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25302⟩⟩, .operator (⟨57493, 1⟩, ⟨57429, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩, (-1)⟩)

def event57500 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25302⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25301⟩⟩) ⟨23166⟩ 57426)

def event57501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25302⟩⟩, .relation 57500 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨23166⟩⟩]⟩, (-1)⟩)

def event57502 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25302⟩⟩, .operator (⟨57493, 0⟩, ⟨57429, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩, (1)⟩)

def exact57503RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨23166⟩⟩]⟩, (-1)⟩]

theorem exact57503RawTermsValid :
    exact57503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25302⟩⟩) exact57503RawTerms .large 57496 (.finite 350212774166528) (some (57498))

def event57504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19244⟩⟩) 0 ⟨12174⟩ 2671

def event57505 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19244⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact57506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩]

theorem exact57506RawTermsValid :
    exact57506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57506 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19244⟩⟩) exact57506RawTerms (.finite 136065468) 57505 .exactZero (none)

def event57507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19246⟩⟩) 0 ⟨19244⟩ 57506

def event57508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19246⟩⟩) 1 ⟨2348⟩ 4

def event57509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19246⟩⟩) (.scale (.predecessor 0 57507 .coefficient) (.value (.predecessor 1 57508 .coefficient)))

def exact57510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩]

theorem exact57510RawTermsValid :
    exact57510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57510 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19246⟩⟩) exact57510RawTerms (.finite 136065468) 57509 .exactZero (none)

def event57511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19247⟩⟩) 0 ⟨5547⟩ 50762

def event57512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19247⟩⟩) 1 ⟨19246⟩ 57510

def event57513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19247⟩⟩) (.product (.predecessor 0 57511 .coefficient) (.predecessor 1 57512 .coefficient) (⟨false, false, none, none, none⟩))

def event57514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19247⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩) [⟨.result 57506 .coefficient, false, none⟩])

def event57515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19247⟩⟩) (.product (.result 50762 .summary) (.transfer 57514) (⟨false, false, none, none, none⟩))

def event57516 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19247⟩⟩, .operator (⟨50762, 0⟩, ⟨57510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩)

def event57517 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19245⟩⟩)

def event57518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event57519 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event57520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event57521 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event57522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event57523 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event57524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event57525 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event57526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 57525

def event57527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 57523

def event57528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 57526 .coefficient) (.value (.predecessor 1 57527 .coefficient)))

def event57529 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event57530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 57529

def event57531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 57521

def event57532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 57530 .coefficient, .predecessor 1 57531 .coefficient])

def event57533 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event57534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 57533

def event57535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 57519

def event57536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 57535 .coefficient))

def event57537 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event57538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11137⟩⟩) 0 ⟨5542⟩ 57537

def event57539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11137⟩⟩) (.authority (.programFamilyFact))

def exact57540RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩], []⟩, (1)⟩]

theorem exact57540RawTermsValid :
    exact57540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11137⟩⟩) exact57540RawTerms (.finite 6) 57539 .exactZero (none)

def event57541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12172⟩⟩) 0 ⟨5542⟩ 57537

def event57542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12172⟩⟩) (.authority (.programFamilyFact))

def exact57543RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩]

theorem exact57543RawTermsValid :
    exact57543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57543 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12172⟩⟩) exact57543RawTerms (.finite 6) 57542 .exactZero (none)

def event57544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 0 ⟨12172⟩ 57543

def event57545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 1 ⟨11137⟩ 57540

def event57546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12173⟩⟩) (.product (.predecessor 0 57544 .coefficient) (.predecessor 1 57545 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12173⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩) [⟨.result 57543 .coefficient, true, some 1⟩, ⟨.result 57540 .coefficient, true, some 1⟩])

def event57548 : Event := .survivorFold (1) 57547

def exact57549RawTerms : List Term := []

theorem exact57549RawTermsValid :
    exact57549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12173⟩⟩) exact57549RawTerms (.finite 36) 57546 (.finite 36) (some (57547))

def event57550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12174⟩⟩) 0 ⟨12173⟩ 57549

def event57551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.identity (.predecessor 0 57550 .coefficient))

def event57552 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.finite 36)

def event57553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19244⟩⟩) 0 ⟨12174⟩ 57552

def event57554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19244⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact57555RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩]

theorem exact57555RawTermsValid :
    exact57555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57555 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19244⟩⟩) exact57555RawTerms (.finite 136065468) 57554 .exactZero (none)

def event57556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact57557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact57557RawTermsValid :
    exact57557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact57557RawTerms .large 57556 .exactZero (none)

def event57558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19245⟩⟩) 0 ⟨6⟩ 57557

def event57559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19245⟩⟩) 1 ⟨19244⟩ 57555

def event57560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19245⟩⟩) (.product (.predecessor 0 57558 .coefficient) (.predecessor 1 57559 .coefficient) (⟨false, false, none, none, none⟩))

def event57561 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19245⟩⟩, .operator (⟨57557, 0⟩, ⟨57555, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩)

def exact57562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩]

theorem exact57562RawTermsValid :
    exact57562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57562 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19245⟩⟩) exact57562RawTerms .large 57560 .exactZero (none)

def event57563 : Event := .preFoldPolynomial 57562 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩] .exactZero none

def exact57564RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩, (1)⟩]

def event57564 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19245⟩⟩) 57563 exact57564RawTerms .large 57560 .exactZero (none)

def event57565 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25305⟩⟩)

def event57566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event57567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event57568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event57569 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event57570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event57571 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event57572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event57573 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event57574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 57573

def event57575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 57571

def event57576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 57574 .coefficient) (.value (.predecessor 1 57575 .coefficient)))

def event57577 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event57578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 57577

def event57579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 57569

def event57580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 57578 .coefficient, .predecessor 1 57579 .coefficient])

def event57581 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event57582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 57581

def event57583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 57567

def event57584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 57583 .coefficient))

def event57585 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event57586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11137⟩⟩) 0 ⟨5542⟩ 57585

def event57587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11137⟩⟩) (.authority (.programFamilyFact))

def exact57588RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩], []⟩, (1)⟩]

theorem exact57588RawTermsValid :
    exact57588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11137⟩⟩) exact57588RawTerms (.finite 6) 57587 .exactZero (none)

def event57589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12172⟩⟩) 0 ⟨5542⟩ 57585

def event57590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12172⟩⟩) (.authority (.programFamilyFact))

def exact57591RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩]

theorem exact57591RawTermsValid :
    exact57591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57591 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12172⟩⟩) exact57591RawTerms (.finite 6) 57590 .exactZero (none)

def event57592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 0 ⟨12172⟩ 57591

def event57593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 1 ⟨11137⟩ 57588

def event57594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12173⟩⟩) (.product (.predecessor 0 57592 .coefficient) (.predecessor 1 57593 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57595 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12173⟩⟩, .operator (⟨57591, 0⟩, ⟨57588, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩)

def exact57596RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩]

theorem exact57596RawTermsValid :
    exact57596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57596 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12173⟩⟩) exact57596RawTerms (.finite 36) 57594 .exactZero (none)

def event57597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12174⟩⟩) 0 ⟨12173⟩ 57596

def event57598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.identity (.predecessor 0 57597 .coefficient))

def event57599 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.finite 36)

def eventLeaf3584 : Array AnnotatedEvent := #[
  { event := event57344
    frameStart := 57292 },
  { event := event57345
    frameStart := 57292 },
  { event := event57346
    frameStart := 57292 },
  { event := event57347
    frameStart := 57292 },
  { event := event57348
    frameStart := 57292 },
  { event := event57349
    frameStart := 57292 },
  { event := event57350
    frameStart := 57292 },
  { event := event57351
    frameStart := 57292 },
  { event := event57352
    frameStart := 57292 },
  { event := event57353
    frameStart := 57292 },
  { event := event57354
    frameStart := 57292 },
  { event := event57355
    frameStart := 57292 },
  { event := event57356
    frameStart := 57292 },
  { event := event57357
    frameStart := 57292 },
  { event := event57358
    frameStart := 57292 },
  { event := event57359
    frameStart := 57292 }
]

def eventLeaf3585 : Array AnnotatedEvent := #[
  { event := event57360
    frameStart := 57292 },
  { event := event57361
    frameStart := 57292 },
  { event := event57362
    frameStart := 57292 },
  { event := event57363
    frameStart := 57292 },
  { event := event57364
    frameStart := 57292 },
  { event := event57365
    frameStart := 57292 },
  { event := event57366
    frameStart := 57292 },
  { event := event57367
    frameStart := 57292 },
  { event := event57368
    frameStart := 57292 },
  { event := event57369
    frameStart := 57292 },
  { event := event57370
    frameStart := 57292 },
  { event := event57371
    frameStart := 57292 },
  { event := event57372
    frameStart := 57292 },
  { event := event57373
    frameStart := 57292 },
  { event := event57374
    frameStart := 57292 },
  { event := event57375
    frameStart := 57292 }
]

def eventLeaf3586 : Array AnnotatedEvent := #[
  { event := event57376
    frameStart := 57292 },
  { event := event57377
    frameStart := 57292 },
  { event := event57378
    frameStart := 57292 },
  { event := event57379
    frameStart := 57292 },
  { event := event57380
    frameStart := 57292 },
  { event := event57381
    frameStart := 57292 },
  { event := event57382
    frameStart := 57292 },
  { event := event57383
    frameStart := 57292 },
  { event := event57384
    frameStart := 57292 },
  { event := event57385
    frameStart := 57292 },
  { event := event57386
    frameStart := 57292 },
  { event := event57387
    frameStart := 57292 },
  { event := event57388
    frameStart := 57292 },
  { event := event57389
    frameStart := 57292 },
  { event := event57390
    frameStart := 57292 },
  { event := event57391
    frameStart := 57292 }
]

def eventLeaf3587 : Array AnnotatedEvent := #[
  { event := event57392
    frameStart := 57292 },
  { event := event57393
    frameStart := 57292 },
  { event := event57394
    frameStart := 57292 },
  { event := event57395
    frameStart := 57292 },
  { event := event57396
    frameStart := 0 },
  { event := event57397
    frameStart := 0 },
  { event := event57398
    frameStart := 0 },
  { event := event57399
    frameStart := 0 },
  { event := event57400
    frameStart := 0 },
  { event := event57401
    frameStart := 0 },
  { event := event57402
    frameStart := 0 },
  { event := event57403
    frameStart := 0 },
  { event := event57404
    frameStart := 0 },
  { event := event57405
    frameStart := 0 },
  { event := event57406
    frameStart := 0 },
  { event := event57407
    frameStart := 0 }
]

def eventLeaf3588 : Array AnnotatedEvent := #[
  { event := event57408
    frameStart := 0 },
  { event := event57409
    frameStart := 0 },
  { event := event57410
    frameStart := 0 },
  { event := event57411
    frameStart := 0 },
  { event := event57412
    frameStart := 0 },
  { event := event57413
    frameStart := 0 },
  { event := event57414
    frameStart := 0 },
  { event := event57415
    frameStart := 0 },
  { event := event57416
    frameStart := 0 },
  { event := event57417
    frameStart := 0 },
  { event := event57418
    frameStart := 0 },
  { event := event57419
    frameStart := 0 },
  { event := event57420
    frameStart := 0 },
  { event := event57421
    frameStart := 0 },
  { event := event57422
    frameStart := 0 },
  { event := event57423
    frameStart := 0 }
]

def eventLeaf3589 : Array AnnotatedEvent := #[
  { event := event57424
    frameStart := 0 },
  { event := event57425
    frameStart := 0 },
  { event := event57426
    frameStart := 0 },
  { event := event57427
    frameStart := 0 },
  { event := event57428
    frameStart := 0 },
  { event := event57429
    frameStart := 0 },
  { event := event57430
    frameStart := 0 },
  { event := event57431
    frameStart := 0 },
  { event := event57432
    frameStart := 0 },
  { event := event57433
    frameStart := 0 },
  { event := event57434
    frameStart := 0 },
  { event := event57435
    frameStart := 0 },
  { event := event57436
    frameStart := 0 },
  { event := event57437
    frameStart := 0 },
  { event := event57438
    frameStart := 0 },
  { event := event57439
    frameStart := 0 }
]

def eventLeaf3590 : Array AnnotatedEvent := #[
  { event := event57440
    frameStart := 0 },
  { event := event57441
    frameStart := 0 },
  { event := event57442
    frameStart := 0 },
  { event := event57443
    frameStart := 0 },
  { event := event57444
    frameStart := 0 },
  { event := event57445
    frameStart := 0 },
  { event := event57446
    frameStart := 0 },
  { event := event57447
    frameStart := 0 },
  { event := event57448
    frameStart := 0 },
  { event := event57449
    frameStart := 0 },
  { event := event57450
    frameStart := 0 },
  { event := event57451
    frameStart := 0 },
  { event := event57452
    frameStart := 0 },
  { event := event57453
    frameStart := 0 },
  { event := event57454
    frameStart := 0 },
  { event := event57455
    frameStart := 0 }
]

def eventLeaf3591 : Array AnnotatedEvent := #[
  { event := event57456
    frameStart := 0 },
  { event := event57457
    frameStart := 0 },
  { event := event57458
    frameStart := 0 },
  { event := event57459
    frameStart := 0 },
  { event := event57460
    frameStart := 0 },
  { event := event57461
    frameStart := 0 },
  { event := event57462
    frameStart := 0 },
  { event := event57463
    frameStart := 0 },
  { event := event57464
    frameStart := 0 },
  { event := event57465
    frameStart := 0 },
  { event := event57466
    frameStart := 0 },
  { event := event57467
    frameStart := 0 },
  { event := event57468
    frameStart := 0 },
  { event := event57469
    frameStart := 0 },
  { event := event57470
    frameStart := 0 },
  { event := event57471
    frameStart := 0 }
]

def eventLeaf3592 : Array AnnotatedEvent := #[
  { event := event57472
    frameStart := 0 },
  { event := event57473
    frameStart := 0 },
  { event := event57474
    frameStart := 0 },
  { event := event57475
    frameStart := 0 },
  { event := event57476
    frameStart := 0 },
  { event := event57477
    frameStart := 0 },
  { event := event57478
    frameStart := 0 },
  { event := event57479
    frameStart := 0 },
  { event := event57480
    frameStart := 0 },
  { event := event57481
    frameStart := 0 },
  { event := event57482
    frameStart := 0 },
  { event := event57483
    frameStart := 0 },
  { event := event57484
    frameStart := 0 },
  { event := event57485
    frameStart := 0 },
  { event := event57486
    frameStart := 0 },
  { event := event57487
    frameStart := 0 }
]

def eventLeaf3593 : Array AnnotatedEvent := #[
  { event := event57488
    frameStart := 0 },
  { event := event57489
    frameStart := 0 },
  { event := event57490
    frameStart := 0 },
  { event := event57491
    frameStart := 0 },
  { event := event57492
    frameStart := 0 },
  { event := event57493
    frameStart := 0 },
  { event := event57494
    frameStart := 0 },
  { event := event57495
    frameStart := 0 },
  { event := event57496
    frameStart := 0 },
  { event := event57497
    frameStart := 0 },
  { event := event57498
    frameStart := 0 },
  { event := event57499
    frameStart := 0 },
  { event := event57500
    frameStart := 0 },
  { event := event57501
    frameStart := 0 },
  { event := event57502
    frameStart := 0 },
  { event := event57503
    frameStart := 0 }
]

def eventLeaf3594 : Array AnnotatedEvent := #[
  { event := event57504
    frameStart := 0 },
  { event := event57505
    frameStart := 0 },
  { event := event57506
    frameStart := 0 },
  { event := event57507
    frameStart := 0 },
  { event := event57508
    frameStart := 0 },
  { event := event57509
    frameStart := 0 },
  { event := event57510
    frameStart := 0 },
  { event := event57511
    frameStart := 0 },
  { event := event57512
    frameStart := 0 },
  { event := event57513
    frameStart := 0 },
  { event := event57514
    frameStart := 0 },
  { event := event57515
    frameStart := 0 },
  { event := event57516
    frameStart := 0 },
  { event := event57517
    frameStart := 57517 },
  { event := event57518
    frameStart := 57517 },
  { event := event57519
    frameStart := 57517 }
]

def eventLeaf3595 : Array AnnotatedEvent := #[
  { event := event57520
    frameStart := 57517 },
  { event := event57521
    frameStart := 57517 },
  { event := event57522
    frameStart := 57517 },
  { event := event57523
    frameStart := 57517 },
  { event := event57524
    frameStart := 57517 },
  { event := event57525
    frameStart := 57517 },
  { event := event57526
    frameStart := 57517 },
  { event := event57527
    frameStart := 57517 },
  { event := event57528
    frameStart := 57517 },
  { event := event57529
    frameStart := 57517 },
  { event := event57530
    frameStart := 57517 },
  { event := event57531
    frameStart := 57517 },
  { event := event57532
    frameStart := 57517 },
  { event := event57533
    frameStart := 57517 },
  { event := event57534
    frameStart := 57517 },
  { event := event57535
    frameStart := 57517 }
]

def eventLeaf3596 : Array AnnotatedEvent := #[
  { event := event57536
    frameStart := 57517 },
  { event := event57537
    frameStart := 57517 },
  { event := event57538
    frameStart := 57517 },
  { event := event57539
    frameStart := 57517 },
  { event := event57540
    frameStart := 57517 },
  { event := event57541
    frameStart := 57517 },
  { event := event57542
    frameStart := 57517 },
  { event := event57543
    frameStart := 57517 },
  { event := event57544
    frameStart := 57517 },
  { event := event57545
    frameStart := 57517 },
  { event := event57546
    frameStart := 57517 },
  { event := event57547
    frameStart := 57517 },
  { event := event57548
    frameStart := 57517 },
  { event := event57549
    frameStart := 57517 },
  { event := event57550
    frameStart := 57517 },
  { event := event57551
    frameStart := 57517 }
]

def eventLeaf3597 : Array AnnotatedEvent := #[
  { event := event57552
    frameStart := 57517 },
  { event := event57553
    frameStart := 57517 },
  { event := event57554
    frameStart := 57517 },
  { event := event57555
    frameStart := 57517 },
  { event := event57556
    frameStart := 57517 },
  { event := event57557
    frameStart := 57517 },
  { event := event57558
    frameStart := 57517 },
  { event := event57559
    frameStart := 57517 },
  { event := event57560
    frameStart := 57517 },
  { event := event57561
    frameStart := 57517 },
  { event := event57562
    frameStart := 57517 },
  { event := event57563
    frameStart := 57517 },
  { event := event57564
    frameStart := 57517 },
  { event := event57565
    frameStart := 57565 },
  { event := event57566
    frameStart := 57565 },
  { event := event57567
    frameStart := 57565 }
]

def eventLeaf3598 : Array AnnotatedEvent := #[
  { event := event57568
    frameStart := 57565 },
  { event := event57569
    frameStart := 57565 },
  { event := event57570
    frameStart := 57565 },
  { event := event57571
    frameStart := 57565 },
  { event := event57572
    frameStart := 57565 },
  { event := event57573
    frameStart := 57565 },
  { event := event57574
    frameStart := 57565 },
  { event := event57575
    frameStart := 57565 },
  { event := event57576
    frameStart := 57565 },
  { event := event57577
    frameStart := 57565 },
  { event := event57578
    frameStart := 57565 },
  { event := event57579
    frameStart := 57565 },
  { event := event57580
    frameStart := 57565 },
  { event := event57581
    frameStart := 57565 },
  { event := event57582
    frameStart := 57565 },
  { event := event57583
    frameStart := 57565 }
]

def eventLeaf3599 : Array AnnotatedEvent := #[
  { event := event57584
    frameStart := 57565 },
  { event := event57585
    frameStart := 57565 },
  { event := event57586
    frameStart := 57565 },
  { event := event57587
    frameStart := 57565 },
  { event := event57588
    frameStart := 57565 },
  { event := event57589
    frameStart := 57565 },
  { event := event57590
    frameStart := 57565 },
  { event := event57591
    frameStart := 57565 },
  { event := event57592
    frameStart := 57565 },
  { event := event57593
    frameStart := 57565 },
  { event := event57594
    frameStart := 57565 },
  { event := event57595
    frameStart := 57565 },
  { event := event57596
    frameStart := 57565 },
  { event := event57597
    frameStart := 57565 },
  { event := event57598
    frameStart := 57565 },
  { event := event57599
    frameStart := 57565 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events224
