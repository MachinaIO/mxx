import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events060

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event15360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 15344

def event15361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 15360 .coefficient))

def event15362 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event15363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10512⟩⟩) 0 ⟨5560⟩ 15362

def event15364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10512⟩⟩) (.authority (.programFamilyFact))

def exact15365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩]

theorem exact15365RawTermsValid :
    exact15365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10512⟩⟩) exact15365RawTerms (.finite 2) 15364 .exactZero (none)

def event15366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9420⟩⟩) 0 ⟨5560⟩ 15362

def event15367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9420⟩⟩) (.authority (.programFamilyFact))

def exact15368RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩], []⟩, (1)⟩]

theorem exact15368RawTermsValid :
    exact15368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9420⟩⟩) exact15368RawTerms (.finite 2) 15367 .exactZero (none)

def event15369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 0 ⟨9420⟩ 15368

def event15370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 1 ⟨10512⟩ 15365

def event15371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10513⟩⟩) (.product (.predecessor 0 15369 .coefficient) (.predecessor 1 15370 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event15372 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10513⟩⟩, .operator (⟨15368, 0⟩, ⟨15365, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩)

def exact15373RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩]

theorem exact15373RawTermsValid :
    exact15373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10513⟩⟩) exact15373RawTerms (.finite 4) 15371 .exactZero (none)

def event15374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10514⟩⟩) 0 ⟨10513⟩ 15373

def event15375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.identity (.predecessor 0 15374 .coefficient))

def event15376 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.finite 4)

def event15377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14808⟩⟩) 0 ⟨10514⟩ 15376

def event15378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14808⟩⟩) (.authority (.programFamilyFact))

def exact15379RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], []⟩, (1)⟩]

theorem exact15379RawTermsValid :
    exact15379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14808⟩⟩) exact15379RawTerms (.finite 2) 15378 .exactZero (none)

def event15380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14809⟩⟩) 0 ⟨14808⟩ 15379

def event15381 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.identity (.predecessor 0 15380 .coefficient))

def event15382 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.finite 2)

def event15383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23731⟩⟩) 0 ⟨14809⟩ 15382

def event15384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23731⟩⟩) (.authority (.programFamilyFact))

def event15385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23731⟩⟩) (.finite 3720)

def event15386 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event15387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23733⟩⟩) 0 ⟨6689⟩ 15386

def event15388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23733⟩⟩) 1 ⟨23731⟩ 15385

def event15389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23733⟩⟩) (.authority (.operator))

def exact15390RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩]

theorem exact15390RawTermsValid :
    exact15390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15390 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23733⟩⟩) exact15390RawTerms .large 15389 .exactZero (none)

def event15391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26406⟩⟩) 0 ⟨23733⟩ 15390

def event15392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26406⟩⟩) (.authority (.operator))

def exact15393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (1)⟩]

theorem exact15393RawTermsValid :
    exact15393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15393 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26406⟩⟩) exact15393RawTerms (.finite 8192) 15392 .exactZero (none)

def event15394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event15395 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event15396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14848⟩⟩) 0 ⟨14809⟩ 15382

def event15397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14848⟩⟩) 1 ⟨110⟩ 15395

def event15398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14848⟩⟩) (.sum [.predecessor 0 15396 .coefficient, .predecessor 1 15397 .coefficient])

def event15399 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14848⟩⟩) (.finite 2)

def event15400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14849⟩⟩) 0 ⟨14848⟩ 15399

def event15401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14849⟩⟩) (.identity (.predecessor 0 15400 .coefficient))

def exact15402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], []⟩, (1)⟩]

theorem exact15402RawTermsValid :
    exact15402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14849⟩⟩) exact15402RawTerms (.finite 2) 15401 .exactZero (none)

def event15403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact15404RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact15404RawTermsValid :
    exact15404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact15404RawTerms .large 15403 .exactZero (none)

def event15405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14850⟩⟩) 0 ⟨6544⟩ 15404

def event15406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14850⟩⟩) 1 ⟨14849⟩ 15402

def event15407 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14850⟩⟩) (.product (.predecessor 0 15405 .coefficient) (.predecessor 1 15406 .coefficient) (⟨false, false, none, none, none⟩))

def event15408 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14850⟩⟩, .operator (⟨15404, 0⟩, ⟨15402, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact15409RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact15409RawTermsValid :
    exact15409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14850⟩⟩) exact15409RawTerms .large 15407 .exactZero (none)

def event15410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 15386

def event15411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact15412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact15412RawTermsValid :
    exact15412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15412 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact15412RawTerms .large 15411 .exactZero (none)

def event15413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14851⟩⟩) 0 ⟨6690⟩ 15412

def event15414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14851⟩⟩) 1 ⟨14850⟩ 15409

def event15415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14851⟩⟩) (.sum [.predecessor 0 15413 .coefficient, .predecessor 1 15414 .coefficient])

def exact15416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15416RawTermsValid :
    exact15416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14851⟩⟩) exact15416RawTerms .large 15415 .exactZero (none)

def event15417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26407⟩⟩) 0 ⟨14851⟩ 15416

def event15418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26407⟩⟩) 1 ⟨26406⟩ 15393

def event15419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26407⟩⟩) (.product (.predecessor 0 15417 .coefficient) (.predecessor 1 15418 .coefficient) (⟨false, false, none, none, none⟩))

def event15420 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26407⟩⟩, .operator (⟨15416, 1⟩, ⟨15393, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩)

def event15421 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26407⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26406⟩⟩) ⟨23733⟩ 15390)

def event15422 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26407⟩⟩, .relation 15421 0, ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (-1)⟩)

def event15423 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26407⟩⟩, .operator (⟨15416, 0⟩, ⟨15393, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (1)⟩)

def exact15424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (-1)⟩]

theorem exact15424RawTermsValid :
    exact15424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26407⟩⟩) exact15424RawTerms .large 15419 .exactZero (none)

def event15425 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15277⟩⟩) 0 ⟨14809⟩ 15382

def event15426 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15277⟩⟩) (.authority (.programFamilyFact))

def exact15427RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], []⟩, (1)⟩]

theorem exact15427RawTermsValid :
    exact15427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15427 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15277⟩⟩) exact15427RawTerms (.finite 43) 15426 .exactZero (none)

def event15428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15278⟩⟩) 0 ⟨6544⟩ 15404

def event15429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15278⟩⟩) 1 ⟨15277⟩ 15427

def event15430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15278⟩⟩) (.product (.predecessor 0 15428 .coefficient) (.predecessor 1 15429 .coefficient) (⟨false, true, none, none, some 1⟩))

def event15431 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15278⟩⟩, .operator (⟨15404, 0⟩, ⟨15427, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact15432RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact15432RawTermsValid :
    exact15432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15432 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15278⟩⟩) exact15432RawTerms .large 15430 .exactZero (none)

def event15433 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 15386

def event15434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact15435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact15435RawTermsValid :
    exact15435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15435 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact15435RawTerms .large 15434 .exactZero (none)

def event15436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15279⟩⟩) 0 ⟨6709⟩ 15435

def event15437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15279⟩⟩) 1 ⟨15278⟩ 15432

def event15438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15279⟩⟩) (.sum [.predecessor 0 15436 .coefficient, .predecessor 1 15437 .coefficient])

def exact15439RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15439RawTermsValid :
    exact15439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15279⟩⟩) exact15439RawTerms .large 15438 .exactZero (none)

def event15440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26410⟩⟩) 0 ⟨15279⟩ 15439

def event15441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26410⟩⟩) 1 ⟨26407⟩ 15424

def event15442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26410⟩⟩) (.sum [.predecessor 0 15440 .coefficient, .predecessor 1 15441 .coefficient])

def exact15443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15443RawTermsValid :
    exact15443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26410⟩⟩) exact15443RawTerms .large 15442 .exactZero (none)

def event15444 : Event := .preFoldPolynomial 15443 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact15445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event15445 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26410⟩⟩) 15444 exact15445RawTerms .large 15442 .exactZero (none)

def event15446 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14809⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨15288, 15446⟩

def event15447 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20411⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20408⟩⟩]⟩) (1) 0 2 (.universal 15446 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20408⟩⟩]⟩) (none) 15445)

def event15448 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20411⟩⟩, .relation 15447 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩)

def event15449 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20411⟩⟩, .relation 15447 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩)

def event15450 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20411⟩⟩, .relation 15447 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event15451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20411⟩⟩, .relation 15447 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def exact15452RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15452RawTermsValid :
    exact15452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15452 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20411⟩⟩) exact15452RawTerms .large 15284 (.finite 1811303510016) (some (15286))

def event15453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26409⟩⟩) 0 ⟨20411⟩ 15452

def event15454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26409⟩⟩) 1 ⟨26408⟩ 15274

def event15455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26409⟩⟩) (.sum [.predecessor 0 15453 .coefficient, .predecessor 1 15454 .coefficient])

def event15456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26409⟩⟩, .operator (⟨15452, 2⟩, ⟨15274, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23733⟩⟩]⟩, (-1)⟩)

def event15457 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26409⟩⟩, .operator (⟨15452, 0⟩, ⟨15274, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩, (1)⟩)

def event15458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26409⟩⟩) (.sum [.result 15452 .summary, .result 15274 .summary])

def exact15459RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15459RawTermsValid :
    exact15459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26409⟩⟩) exact15459RawTerms .large 15455 (.finite 1291889174379421642752) (some (15458))

def event15460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26620⟩⟩) 0 ⟨26409⟩ 15459

def event15461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26620⟩⟩) 1 ⟨26619⟩ 14958

def event15462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26620⟩⟩) (.sum [.predecessor 0 15460 .coefficient, .predecessor 1 15461 .coefficient])

def event15463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26620⟩⟩) (.sum [.result 15459 .summary, .result 14958 .summary])

def exact15464RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15464RawTermsValid :
    exact15464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26620⟩⟩) exact15464RawTerms .large 15462 (.finite 2583789554981353578496) (some (15463))

def event15465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26837⟩⟩) 0 ⟨26620⟩ 15464

def event15466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26837⟩⟩) 1 ⟨26836⟩ 14457

def event15467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26837⟩⟩) (.sum [.predecessor 0 15465 .coefficient, .predecessor 1 15466 .coefficient])

def event15468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26837⟩⟩) (.sum [.result 15464 .summary, .result 14457 .summary])

def exact15469RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15469RawTermsValid :
    exact15469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26837⟩⟩) exact15469RawTerms .large 15467 (.finite 3875701141805795807232) (some (15468))

def event15470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27054⟩⟩) 0 ⟨26837⟩ 15469

def event15471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27054⟩⟩) 1 ⟨27053⟩ 13956

def event15472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27054⟩⟩) (.sum [.predecessor 0 15470 .coefficient, .predecessor 1 15471 .coefficient])

def event15473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27054⟩⟩) (.sum [.result 15469 .summary, .result 13956 .summary])

def exact15474RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15474RawTermsValid :
    exact15474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27054⟩⟩) exact15474RawTerms .large 15472 (.finite 5167635141075258621952) (some (15473))

def event15475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27271⟩⟩) 0 ⟨27054⟩ 15474

def event15476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27271⟩⟩) 1 ⟨27270⟩ 13455

def event15477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27271⟩⟩) (.sum [.predecessor 0 15475 .coefficient, .predecessor 1 15476 .coefficient])

def event15478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27271⟩⟩) (.sum [.result 15474 .summary, .result 13455 .summary])

def exact15479RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15479RawTermsValid :
    exact15479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27271⟩⟩) exact15479RawTerms .large 15477 (.finite 6459613965234762608640) (some (15478))

def event15480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27488⟩⟩) 0 ⟨27271⟩ 15479

def event15481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27488⟩⟩) 1 ⟨27487⟩ 12954

def event15482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27488⟩⟩) (.sum [.predecessor 0 15480 .coefficient, .predecessor 1 15481 .coefficient])

def event15483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27488⟩⟩) (.sum [.result 15479 .summary, .result 12954 .summary])

def exact15484RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15484RawTermsValid :
    exact15484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27488⟩⟩) exact15484RawTerms .large 15482 (.finite 7751615201839287181312) (some (15483))

def event15485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27705⟩⟩) 0 ⟨27488⟩ 15484

def event15486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27705⟩⟩) 1 ⟨27704⟩ 12453

def event15487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27705⟩⟩) (.sum [.predecessor 0 15485 .coefficient, .predecessor 1 15486 .coefficient])

def event15488 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27705⟩⟩) (.sum [.result 15484 .summary, .result 12453 .summary])

def exact15489RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15489RawTermsValid :
    exact15489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15489 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27705⟩⟩) exact15489RawTerms .large 15487 (.finite 9043661263333852925952) (some (15488))

def event15490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27922⟩⟩) 0 ⟨27705⟩ 15489

def event15491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27922⟩⟩) 1 ⟨27921⟩ 11952

def event15492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27922⟩⟩) (.sum [.predecessor 0 15490 .coefficient, .predecessor 1 15491 .coefficient])

def event15493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27922⟩⟩) (.sum [.result 15489 .summary, .result 11952 .summary])

def exact15494RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15494RawTermsValid :
    exact15494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27922⟩⟩) exact15494RawTerms .large 15492 (.finite 10335729737273439256576) (some (15493))

def event15495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28139⟩⟩) 0 ⟨27922⟩ 15494

def event15496 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28139⟩⟩) 1 ⟨28138⟩ 11451

def event15497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28139⟩⟩) (.sum [.predecessor 0 15495 .coefficient, .predecessor 1 15496 .coefficient])

def event15498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28139⟩⟩) (.sum [.result 15494 .summary, .result 11451 .summary])

def exact15499RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15499RawTermsValid :
    exact15499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15499 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28139⟩⟩) exact15499RawTerms .large 15497 (.finite 11627843036103066759168) (some (15498))

def event15500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28356⟩⟩) 0 ⟨28139⟩ 15499

def event15501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28356⟩⟩) 1 ⟨28355⟩ 10950

def event15502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28356⟩⟩) (.sum [.predecessor 0 15500 .coefficient, .predecessor 1 15501 .coefficient])

def event15503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28356⟩⟩) (.sum [.result 15499 .summary, .result 10950 .summary])

def exact15504RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15504RawTermsValid :
    exact15504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28356⟩⟩) exact15504RawTerms .large 15502 (.finite 12920023572267756019712) (some (15503))

def event15505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28573⟩⟩) 0 ⟨28356⟩ 15504

def event15506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28573⟩⟩) 1 ⟨28572⟩ 10449

def event15507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28573⟩⟩) (.sum [.predecessor 0 15505 .coefficient, .predecessor 1 15506 .coefficient])

def event15508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28573⟩⟩) (.sum [.result 15504 .summary, .result 10449 .summary])

def exact15509RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15509RawTermsValid :
    exact15509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28573⟩⟩) exact15509RawTerms .large 15507 (.finite 14212226520877465866240) (some (15508))

def event15510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28790⟩⟩) 0 ⟨28573⟩ 15509

def event15511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28790⟩⟩) 1 ⟨28789⟩ 9948

def event15512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28790⟩⟩) (.sum [.predecessor 0 15510 .coefficient, .predecessor 1 15511 .coefficient])

def event15513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28790⟩⟩) (.sum [.result 15509 .summary, .result 9948 .summary])

def exact15514RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15514RawTermsValid :
    exact15514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15514 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28790⟩⟩) exact15514RawTerms .large 15512 (.finite 15504496706822237470720) (some (15513))

def event15515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29007⟩⟩) 0 ⟨28790⟩ 15514

def event15516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29007⟩⟩) 1 ⟨29006⟩ 9447

def event15517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29007⟩⟩) (.sum [.predecessor 0 15515 .coefficient, .predecessor 1 15516 .coefficient])

def event15518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29007⟩⟩) (.sum [.result 15514 .summary, .result 9447 .summary])

def exact15519RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15519RawTermsValid :
    exact15519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15519 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29007⟩⟩) exact15519RawTerms .large 15517 (.finite 16796811717657050247168) (some (15518))

def event15520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29224⟩⟩) 0 ⟨29007⟩ 15519

def event15521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29224⟩⟩) 1 ⟨29223⟩ 8946

def event15522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29224⟩⟩) (.sum [.predecessor 0 15520 .coefficient, .predecessor 1 15521 .coefficient])

def event15523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29224⟩⟩) (.sum [.result 15519 .summary, .result 8946 .summary])

def exact15524RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15524RawTermsValid :
    exact15524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15524 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29224⟩⟩) exact15524RawTerms .large 15522 (.finite 18089149140936883609600) (some (15523))

def event15525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29441⟩⟩) 0 ⟨29224⟩ 15524

def event15526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29441⟩⟩) 1 ⟨29440⟩ 8445

def event15527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29441⟩⟩) (.sum [.predecessor 0 15525 .coefficient, .predecessor 1 15526 .coefficient])

def event15528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29441⟩⟩) (.sum [.result 15524 .summary, .result 8445 .summary])

def exact15529RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15529RawTermsValid :
    exact15529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15529 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29441⟩⟩) exact15529RawTerms .large 15527 (.finite 19381531389106758144000) (some (15528))

def event15530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29658⟩⟩) 0 ⟨29441⟩ 15529

def event15531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29658⟩⟩) 1 ⟨29657⟩ 7944

def event15532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29658⟩⟩) (.sum [.predecessor 0 15530 .coefficient, .predecessor 1 15531 .coefficient])

def event15533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29658⟩⟩) (.sum [.result 15529 .summary, .result 7944 .summary])

def exact15534RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15534RawTermsValid :
    exact15534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29658⟩⟩) exact15534RawTerms .large 15532 (.finite 20673980874611694436352) (some (15533))

def event15535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29875⟩⟩) 0 ⟨29658⟩ 15534

def event15536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29875⟩⟩) 1 ⟨29874⟩ 7443

def event15537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29875⟩⟩) (.sum [.predecessor 0 15535 .coefficient, .predecessor 1 15536 .coefficient])

def event15538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29875⟩⟩) (.sum [.result 15534 .summary, .result 7443 .summary])

def exact15539RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15539RawTermsValid :
    exact15539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15539 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29875⟩⟩) exact15539RawTerms .large 15537 (.finite 21966497597451692486656) (some (15538))

def event15540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30209⟩⟩) 0 ⟨29875⟩ 15539

def event15541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30209⟩⟩) 1 ⟨30208⟩ 6942

def event15542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30209⟩⟩) (.sum [.predecessor 0 15540 .coefficient, .predecessor 1 15541 .coefficient])

def event15543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30209⟩⟩) (.sum [.result 15539 .summary, .result 6942 .summary])

def exact15544RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact15544RawTermsValid :
    exact15544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event15544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30209⟩⟩) exact15544RawTerms .large 15542 (.finite 23259036732736711122944) (some (15543))

def event15545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30210⟩⟩) 0 ⟨30209⟩ 15544

def event15546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30210⟩⟩) 1 ⟨18693⟩ 6419

def event15547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30210⟩⟩) (.product (.predecessor 0 15545 .coefficient) (.predecessor 1 15546 .coefficient) (⟨false, false, none, none, none⟩))

def event15548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30210⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) [⟨.result 6419 .coefficient, false, none⟩])

def event15549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30210⟩⟩) (.product (.result 15544 .summary) (.transfer 15548) (⟨false, false, none, none, none⟩))

def event15550 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 33⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15551 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15552 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15551 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15553 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 17⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15554 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 29⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15555 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15556 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15555 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15557 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 16⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15558 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 28⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15559 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15560 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15559 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15561 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 15⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15562 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 27⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15563 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15564 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15563 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15565 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 14⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15566 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 34⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15567 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15568 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15567 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15569 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 13⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15570 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 32⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15571 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15572 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15571 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 12⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15574 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 30⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15575 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15576 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15575 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15577 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 11⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15578 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 26⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15579 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15580 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15579 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15581 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 10⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15582 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 35⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15583 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15584 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15583 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15585 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 9⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15586 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 25⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15587 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15588 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15587 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15589 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 8⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15590 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 24⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15591 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15592 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15591 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15593 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 7⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15594 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 23⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15595 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15595 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15597 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 6⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15598 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 22⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15599 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15600 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15599 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15601 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 5⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15602 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 21⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15603 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15604 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15603 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15605 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 4⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15606 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 31⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15607 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15608 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15607 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15609 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 3⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15610 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 20⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15611 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def event15612 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .relation 15611 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event15613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 2⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event15614 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30210⟩⟩, .operator (⟨15544, 19⟩, ⟨6419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event15615 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416)

def eventLeaf960 : Array AnnotatedEvent := #[
  { event := event15360
    frameStart := 15342 },
  { event := event15361
    frameStart := 15342 },
  { event := event15362
    frameStart := 15342 },
  { event := event15363
    frameStart := 15342 },
  { event := event15364
    frameStart := 15342 },
  { event := event15365
    frameStart := 15342 },
  { event := event15366
    frameStart := 15342 },
  { event := event15367
    frameStart := 15342 },
  { event := event15368
    frameStart := 15342 },
  { event := event15369
    frameStart := 15342 },
  { event := event15370
    frameStart := 15342 },
  { event := event15371
    frameStart := 15342 },
  { event := event15372
    frameStart := 15342 },
  { event := event15373
    frameStart := 15342 },
  { event := event15374
    frameStart := 15342 },
  { event := event15375
    frameStart := 15342 }
]

def eventLeaf961 : Array AnnotatedEvent := #[
  { event := event15376
    frameStart := 15342 },
  { event := event15377
    frameStart := 15342 },
  { event := event15378
    frameStart := 15342 },
  { event := event15379
    frameStart := 15342 },
  { event := event15380
    frameStart := 15342 },
  { event := event15381
    frameStart := 15342 },
  { event := event15382
    frameStart := 15342 },
  { event := event15383
    frameStart := 15342 },
  { event := event15384
    frameStart := 15342 },
  { event := event15385
    frameStart := 15342 },
  { event := event15386
    frameStart := 15342 },
  { event := event15387
    frameStart := 15342 },
  { event := event15388
    frameStart := 15342 },
  { event := event15389
    frameStart := 15342 },
  { event := event15390
    frameStart := 15342 },
  { event := event15391
    frameStart := 15342 }
]

def eventLeaf962 : Array AnnotatedEvent := #[
  { event := event15392
    frameStart := 15342 },
  { event := event15393
    frameStart := 15342 },
  { event := event15394
    frameStart := 15342 },
  { event := event15395
    frameStart := 15342 },
  { event := event15396
    frameStart := 15342 },
  { event := event15397
    frameStart := 15342 },
  { event := event15398
    frameStart := 15342 },
  { event := event15399
    frameStart := 15342 },
  { event := event15400
    frameStart := 15342 },
  { event := event15401
    frameStart := 15342 },
  { event := event15402
    frameStart := 15342 },
  { event := event15403
    frameStart := 15342 },
  { event := event15404
    frameStart := 15342 },
  { event := event15405
    frameStart := 15342 },
  { event := event15406
    frameStart := 15342 },
  { event := event15407
    frameStart := 15342 }
]

def eventLeaf963 : Array AnnotatedEvent := #[
  { event := event15408
    frameStart := 15342 },
  { event := event15409
    frameStart := 15342 },
  { event := event15410
    frameStart := 15342 },
  { event := event15411
    frameStart := 15342 },
  { event := event15412
    frameStart := 15342 },
  { event := event15413
    frameStart := 15342 },
  { event := event15414
    frameStart := 15342 },
  { event := event15415
    frameStart := 15342 },
  { event := event15416
    frameStart := 15342 },
  { event := event15417
    frameStart := 15342 },
  { event := event15418
    frameStart := 15342 },
  { event := event15419
    frameStart := 15342 },
  { event := event15420
    frameStart := 15342 },
  { event := event15421
    frameStart := 15342 },
  { event := event15422
    frameStart := 15342 },
  { event := event15423
    frameStart := 15342 }
]

def eventLeaf964 : Array AnnotatedEvent := #[
  { event := event15424
    frameStart := 15342 },
  { event := event15425
    frameStart := 15342 },
  { event := event15426
    frameStart := 15342 },
  { event := event15427
    frameStart := 15342 },
  { event := event15428
    frameStart := 15342 },
  { event := event15429
    frameStart := 15342 },
  { event := event15430
    frameStart := 15342 },
  { event := event15431
    frameStart := 15342 },
  { event := event15432
    frameStart := 15342 },
  { event := event15433
    frameStart := 15342 },
  { event := event15434
    frameStart := 15342 },
  { event := event15435
    frameStart := 15342 },
  { event := event15436
    frameStart := 15342 },
  { event := event15437
    frameStart := 15342 },
  { event := event15438
    frameStart := 15342 },
  { event := event15439
    frameStart := 15342 }
]

def eventLeaf965 : Array AnnotatedEvent := #[
  { event := event15440
    frameStart := 15342 },
  { event := event15441
    frameStart := 15342 },
  { event := event15442
    frameStart := 15342 },
  { event := event15443
    frameStart := 15342 },
  { event := event15444
    frameStart := 15342 },
  { event := event15445
    frameStart := 15342 },
  { event := event15446
    frameStart := 0 },
  { event := event15447
    frameStart := 0 },
  { event := event15448
    frameStart := 0 },
  { event := event15449
    frameStart := 0 },
  { event := event15450
    frameStart := 0 },
  { event := event15451
    frameStart := 0 },
  { event := event15452
    frameStart := 0 },
  { event := event15453
    frameStart := 0 },
  { event := event15454
    frameStart := 0 },
  { event := event15455
    frameStart := 0 }
]

def eventLeaf966 : Array AnnotatedEvent := #[
  { event := event15456
    frameStart := 0 },
  { event := event15457
    frameStart := 0 },
  { event := event15458
    frameStart := 0 },
  { event := event15459
    frameStart := 0 },
  { event := event15460
    frameStart := 0 },
  { event := event15461
    frameStart := 0 },
  { event := event15462
    frameStart := 0 },
  { event := event15463
    frameStart := 0 },
  { event := event15464
    frameStart := 0 },
  { event := event15465
    frameStart := 0 },
  { event := event15466
    frameStart := 0 },
  { event := event15467
    frameStart := 0 },
  { event := event15468
    frameStart := 0 },
  { event := event15469
    frameStart := 0 },
  { event := event15470
    frameStart := 0 },
  { event := event15471
    frameStart := 0 }
]

def eventLeaf967 : Array AnnotatedEvent := #[
  { event := event15472
    frameStart := 0 },
  { event := event15473
    frameStart := 0 },
  { event := event15474
    frameStart := 0 },
  { event := event15475
    frameStart := 0 },
  { event := event15476
    frameStart := 0 },
  { event := event15477
    frameStart := 0 },
  { event := event15478
    frameStart := 0 },
  { event := event15479
    frameStart := 0 },
  { event := event15480
    frameStart := 0 },
  { event := event15481
    frameStart := 0 },
  { event := event15482
    frameStart := 0 },
  { event := event15483
    frameStart := 0 },
  { event := event15484
    frameStart := 0 },
  { event := event15485
    frameStart := 0 },
  { event := event15486
    frameStart := 0 },
  { event := event15487
    frameStart := 0 }
]

def eventLeaf968 : Array AnnotatedEvent := #[
  { event := event15488
    frameStart := 0 },
  { event := event15489
    frameStart := 0 },
  { event := event15490
    frameStart := 0 },
  { event := event15491
    frameStart := 0 },
  { event := event15492
    frameStart := 0 },
  { event := event15493
    frameStart := 0 },
  { event := event15494
    frameStart := 0 },
  { event := event15495
    frameStart := 0 },
  { event := event15496
    frameStart := 0 },
  { event := event15497
    frameStart := 0 },
  { event := event15498
    frameStart := 0 },
  { event := event15499
    frameStart := 0 },
  { event := event15500
    frameStart := 0 },
  { event := event15501
    frameStart := 0 },
  { event := event15502
    frameStart := 0 },
  { event := event15503
    frameStart := 0 }
]

def eventLeaf969 : Array AnnotatedEvent := #[
  { event := event15504
    frameStart := 0 },
  { event := event15505
    frameStart := 0 },
  { event := event15506
    frameStart := 0 },
  { event := event15507
    frameStart := 0 },
  { event := event15508
    frameStart := 0 },
  { event := event15509
    frameStart := 0 },
  { event := event15510
    frameStart := 0 },
  { event := event15511
    frameStart := 0 },
  { event := event15512
    frameStart := 0 },
  { event := event15513
    frameStart := 0 },
  { event := event15514
    frameStart := 0 },
  { event := event15515
    frameStart := 0 },
  { event := event15516
    frameStart := 0 },
  { event := event15517
    frameStart := 0 },
  { event := event15518
    frameStart := 0 },
  { event := event15519
    frameStart := 0 }
]

def eventLeaf970 : Array AnnotatedEvent := #[
  { event := event15520
    frameStart := 0 },
  { event := event15521
    frameStart := 0 },
  { event := event15522
    frameStart := 0 },
  { event := event15523
    frameStart := 0 },
  { event := event15524
    frameStart := 0 },
  { event := event15525
    frameStart := 0 },
  { event := event15526
    frameStart := 0 },
  { event := event15527
    frameStart := 0 },
  { event := event15528
    frameStart := 0 },
  { event := event15529
    frameStart := 0 },
  { event := event15530
    frameStart := 0 },
  { event := event15531
    frameStart := 0 },
  { event := event15532
    frameStart := 0 },
  { event := event15533
    frameStart := 0 },
  { event := event15534
    frameStart := 0 },
  { event := event15535
    frameStart := 0 }
]

def eventLeaf971 : Array AnnotatedEvent := #[
  { event := event15536
    frameStart := 0 },
  { event := event15537
    frameStart := 0 },
  { event := event15538
    frameStart := 0 },
  { event := event15539
    frameStart := 0 },
  { event := event15540
    frameStart := 0 },
  { event := event15541
    frameStart := 0 },
  { event := event15542
    frameStart := 0 },
  { event := event15543
    frameStart := 0 },
  { event := event15544
    frameStart := 0 },
  { event := event15545
    frameStart := 0 },
  { event := event15546
    frameStart := 0 },
  { event := event15547
    frameStart := 0 },
  { event := event15548
    frameStart := 0 },
  { event := event15549
    frameStart := 0 },
  { event := event15550
    frameStart := 0 },
  { event := event15551
    frameStart := 0 }
]

def eventLeaf972 : Array AnnotatedEvent := #[
  { event := event15552
    frameStart := 0 },
  { event := event15553
    frameStart := 0 },
  { event := event15554
    frameStart := 0 },
  { event := event15555
    frameStart := 0 },
  { event := event15556
    frameStart := 0 },
  { event := event15557
    frameStart := 0 },
  { event := event15558
    frameStart := 0 },
  { event := event15559
    frameStart := 0 },
  { event := event15560
    frameStart := 0 },
  { event := event15561
    frameStart := 0 },
  { event := event15562
    frameStart := 0 },
  { event := event15563
    frameStart := 0 },
  { event := event15564
    frameStart := 0 },
  { event := event15565
    frameStart := 0 },
  { event := event15566
    frameStart := 0 },
  { event := event15567
    frameStart := 0 }
]

def eventLeaf973 : Array AnnotatedEvent := #[
  { event := event15568
    frameStart := 0 },
  { event := event15569
    frameStart := 0 },
  { event := event15570
    frameStart := 0 },
  { event := event15571
    frameStart := 0 },
  { event := event15572
    frameStart := 0 },
  { event := event15573
    frameStart := 0 },
  { event := event15574
    frameStart := 0 },
  { event := event15575
    frameStart := 0 },
  { event := event15576
    frameStart := 0 },
  { event := event15577
    frameStart := 0 },
  { event := event15578
    frameStart := 0 },
  { event := event15579
    frameStart := 0 },
  { event := event15580
    frameStart := 0 },
  { event := event15581
    frameStart := 0 },
  { event := event15582
    frameStart := 0 },
  { event := event15583
    frameStart := 0 }
]

def eventLeaf974 : Array AnnotatedEvent := #[
  { event := event15584
    frameStart := 0 },
  { event := event15585
    frameStart := 0 },
  { event := event15586
    frameStart := 0 },
  { event := event15587
    frameStart := 0 },
  { event := event15588
    frameStart := 0 },
  { event := event15589
    frameStart := 0 },
  { event := event15590
    frameStart := 0 },
  { event := event15591
    frameStart := 0 },
  { event := event15592
    frameStart := 0 },
  { event := event15593
    frameStart := 0 },
  { event := event15594
    frameStart := 0 },
  { event := event15595
    frameStart := 0 },
  { event := event15596
    frameStart := 0 },
  { event := event15597
    frameStart := 0 },
  { event := event15598
    frameStart := 0 },
  { event := event15599
    frameStart := 0 }
]

def eventLeaf975 : Array AnnotatedEvent := #[
  { event := event15600
    frameStart := 0 },
  { event := event15601
    frameStart := 0 },
  { event := event15602
    frameStart := 0 },
  { event := event15603
    frameStart := 0 },
  { event := event15604
    frameStart := 0 },
  { event := event15605
    frameStart := 0 },
  { event := event15606
    frameStart := 0 },
  { event := event15607
    frameStart := 0 },
  { event := event15608
    frameStart := 0 },
  { event := event15609
    frameStart := 0 },
  { event := event15610
    frameStart := 0 },
  { event := event15611
    frameStart := 0 },
  { event := event15612
    frameStart := 0 },
  { event := event15613
    frameStart := 0 },
  { event := event15614
    frameStart := 0 },
  { event := event15615
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events060
