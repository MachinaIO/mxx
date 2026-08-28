import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events228

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event58368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact58369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58369RawTermsValid :
    exact58369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact58369RawTerms .large 58368 .exactZero (none)

def event58370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30480⟩⟩) 0 ⟨6908⟩ 58369

def event58371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30480⟩⟩) 1 ⟨30479⟩ 58367

def event58372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30480⟩⟩) (.product (.predecessor 0 58370 .coefficient) (.predecessor 1 58371 .coefficient) (⟨false, false, none, none, none⟩))

def event58373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30480⟩⟩, .operator (⟨58369, 0⟩, ⟨58367, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact58374RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58374RawTermsValid :
    exact58374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30480⟩⟩) exact58374RawTerms .large 58372 .exactZero (none)

def event58375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 58351

def event58376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact58377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact58377RawTermsValid :
    exact58377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58377 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact58377RawTerms .large 58376 .exactZero (none)

def event58378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30481⟩⟩) 0 ⟨7190⟩ 58377

def event58379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30481⟩⟩) 1 ⟨30480⟩ 58374

def event58380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30481⟩⟩) (.sum [.predecessor 0 58378 .coefficient, .predecessor 1 58379 .coefficient])

def exact58381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58381RawTermsValid :
    exact58381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58381 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30481⟩⟩) exact58381RawTerms .large 58380 .exactZero (none)

def event58382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31164⟩⟩) 0 ⟨30481⟩ 58381

def event58383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31164⟩⟩) 1 ⟨31163⟩ 58358

def event58384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31164⟩⟩) (.product (.predecessor 0 58382 .coefficient) (.predecessor 1 58383 .coefficient) (⟨false, false, none, none, none⟩))

def event58385 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31164⟩⟩, .operator (⟨58381, 0⟩, ⟨58358, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (1)⟩)

def event58386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31164⟩⟩, .operator (⟨58381, 1⟩, ⟨58358, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩)

def event58387 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31164⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31163⟩⟩) ⟨30312⟩ 58355)

def event58388 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31164⟩⟩, .relation 58387 0, ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (-1)⟩)

def exact58389RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (-1)⟩]

theorem exact58389RawTermsValid :
    exact58389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58389 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31164⟩⟩) exact58389RawTerms .large 58384 .exactZero (none)

def event58390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29406⟩⟩) 0 ⟨29153⟩ 58347

def event58391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29406⟩⟩) (.authority (.programFamilyFact))

def exact58392RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], []⟩, (1)⟩]

theorem exact58392RawTermsValid :
    exact58392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29406⟩⟩) exact58392RawTerms (.finite 36) 58391 .exactZero (none)

def event58393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29408⟩⟩) 0 ⟨6908⟩ 58369

def event58394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29408⟩⟩) 1 ⟨29406⟩ 58392

def event58395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29408⟩⟩) (.product (.predecessor 0 58393 .coefficient) (.predecessor 1 58394 .coefficient) (⟨false, true, none, none, some 1⟩))

def event58396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29408⟩⟩, .operator (⟨58369, 0⟩, ⟨58392, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact58397RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58397RawTermsValid :
    exact58397RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58397 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29408⟩⟩) exact58397RawTerms .large 58395 .exactZero (none)

def event58398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7219⟩⟩) 0 ⟨7177⟩ 58351

def event58399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7219⟩⟩) (.authority (.operator))

def exact58400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩]

theorem exact58400RawTermsValid :
    exact58400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7219⟩⟩) exact58400RawTerms .large 58399 .exactZero (none)

def event58401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29409⟩⟩) 0 ⟨7219⟩ 58400

def event58402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29409⟩⟩) 1 ⟨29408⟩ 58397

def event58403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29409⟩⟩) (.sum [.predecessor 0 58401 .coefficient, .predecessor 1 58402 .coefficient])

def exact58404RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58404RawTermsValid :
    exact58404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58404 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29409⟩⟩) exact58404RawTerms .large 58403 .exactZero (none)

def event58405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31168⟩⟩) 0 ⟨29409⟩ 58404

def event58406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31168⟩⟩) 1 ⟨31164⟩ 58389

def event58407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31168⟩⟩) (.sum [.predecessor 0 58405 .coefficient, .predecessor 1 58406 .coefficient])

def exact58408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58408RawTermsValid :
    exact58408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31168⟩⟩) exact58408RawTerms .large 58407 .exactZero (none)

def event58409 : Event := .preFoldPolynomial 58408 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact58410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event58410 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨31168⟩⟩) 58409 exact58410RawTerms .large 58407 .exactZero (none)

def event58411 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨29153⟩⟩) ⟨⟨98⟩, ⟨80⟩, ⟨135⟩⟩ ⟨58253, 58411⟩

def event58412 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29995⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (1) 0 2 (.universal 58411 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29992⟩⟩]⟩) (none) 58410)

def event58413 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29995⟩⟩, .relation 58412 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩)

def event58414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29995⟩⟩, .relation 58412 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩)

def event58415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29995⟩⟩, .relation 58412 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (1)⟩)

def event58416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29995⟩⟩, .relation 58412 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact58417RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58417RawTermsValid :
    exact58417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58417 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29995⟩⟩) exact58417RawTerms .large 58249 (.finite 202072841853861888) (some (58251))

def event58418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31166⟩⟩) 0 ⟨29995⟩ 58417

def event58419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31166⟩⟩) 1 ⟨31165⟩ 58239

def event58420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31166⟩⟩) (.sum [.predecessor 0 58418 .coefficient, .predecessor 1 58419 .coefficient])

def event58421 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31166⟩⟩, .operator (⟨58417, 0⟩, ⟨58239, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31163⟩⟩]⟩, (1)⟩)

def event58422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31166⟩⟩, .operator (⟨58417, 2⟩, ⟨58239, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨30312⟩⟩]⟩, (-1)⟩)

def event58423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31166⟩⟩) (.sum [.result 58417 .summary, .result 58239 .summary])

def exact58424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58424RawTermsValid :
    exact58424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31166⟩⟩) exact58424RawTerms .large 58420 (.finite 32192146870060392302605751287808) (some (58423))

def event58425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31167⟩⟩) 0 ⟨31166⟩ 58424

def event58426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31167⟩⟩) 1 ⟨7168⟩ 15662

def event58427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31167⟩⟩) (.product (.predecessor 0 58425 .coefficient) (.predecessor 1 58426 .coefficient) (⟨false, false, none, none, none⟩))

def event58428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31167⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) [⟨.result 15658 .coefficient, false, none⟩])

def event58429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31167⟩⟩) (.product (.result 58424 .summary) (.transfer 58428) (⟨false, false, none, none, none⟩))

def event58430 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31167⟩⟩, .operator (⟨58424, 0⟩, ⟨15662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def event58431 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31167⟩⟩, .operator (⟨58424, 1⟩, ⟨15662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event58432 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31167⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7167⟩⟩) ⟨7049⟩ 15655)

def event58433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31167⟩⟩, .relation 58432 0, ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact58434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩]

theorem exact58434RawTermsValid :
    exact58434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31167⟩⟩) exact58434RawTerms .large 58427 (.finite 345660544987345366211554593406613108817920) (some (58429))

def event58435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27632⟩⟩) 0 ⟨7177⟩ 15500

def event58436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27632⟩⟩) 1 ⟨27631⟩ 50021

def event58437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27632⟩⟩) (.authority (.operator))

def exact58438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (1)⟩]

theorem exact58438RawTermsValid :
    exact58438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27632⟩⟩) exact58438RawTerms .large 58437 .exactZero (none)

def event58439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28483⟩⟩) 0 ⟨27632⟩ 58438

def event58440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28483⟩⟩) (.authority (.operator))

def exact58441RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩]

theorem exact58441RawTermsValid :
    exact58441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58441 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28483⟩⟩) exact58441RawTerms (.finite 8192) 58440 .exactZero (none)

def event58442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28485⟩⟩) 0 ⟨28009⟩ 50305

def event58443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28485⟩⟩) 1 ⟨28483⟩ 58441

def event58444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28485⟩⟩) (.product (.predecessor 0 58442 .coefficient) (.predecessor 1 58443 .coefficient) (⟨false, false, none, none, none⟩))

def event58445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28485⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩) [⟨.result 58441 .coefficient, false, none⟩])

def event58446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28485⟩⟩) (.product (.result 50305 .summary) (.transfer 58445) (⟨false, false, none, none, none⟩))

def event58447 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28485⟩⟩, .operator (⟨50305, 0⟩, ⟨58441, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩)

def event58448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28485⟩⟩, .operator (⟨50305, 1⟩, ⟨58441, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (-1)⟩)

def event58449 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28485⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28483⟩⟩) ⟨27632⟩ 58438)

def event58450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28485⟩⟩, .relation 58449 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (-1)⟩)

def exact58451RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (-1)⟩]

theorem exact58451RawTermsValid :
    exact58451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58451 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28485⟩⟩) exact58451RawTerms .large 58444 (.finite 32191557518723128098041228165120) (some (58446))

def event58452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27312⟩⟩) 0 ⟨26473⟩ 1768

def event58453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27312⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact58454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩]

theorem exact58454RawTermsValid :
    exact58454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27312⟩⟩) exact58454RawTerms (.finite 5647228698) 58453 .exactZero (none)

def event58455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27314⟩⟩) 0 ⟨27312⟩ 58454

def event58456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27314⟩⟩) 1 ⟨2370⟩ 4

def event58457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27314⟩⟩) (.scale (.predecessor 0 58455 .coefficient) (.value (.predecessor 1 58456 .coefficient)))

def exact58458RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩]

theorem exact58458RawTermsValid :
    exact58458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27314⟩⟩) exact58458RawTerms (.finite 5647228698) 58457 .exactZero (none)

def event58459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27315⟩⟩) 0 ⟨11216⟩ 46745

def event58460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27315⟩⟩) 1 ⟨27314⟩ 58458

def event58461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27315⟩⟩) (.product (.predecessor 0 58459 .coefficient) (.predecessor 1 58460 .coefficient) (⟨false, false, none, none, none⟩))

def event58462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27315⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩) [⟨.result 58454 .coefficient, false, none⟩])

def event58463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27315⟩⟩) (.product (.result 46745 .summary) (.transfer 58462) (⟨false, false, none, none, none⟩))

def event58464 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27315⟩⟩, .operator (⟨46745, 0⟩, ⟨58458, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩)

def event58465 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27313⟩⟩)

def event58466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event58467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event58468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event58469 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event58470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event58471 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event58472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event58473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event58474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 58473

def event58475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 58471

def event58476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 58474 .coefficient) (.value (.predecessor 1 58475 .coefficient)))

def event58477 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event58478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 58477

def event58479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 58469

def event58480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 58478 .coefficient, .predecessor 1 58479 .coefficient])

def event58481 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event58482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 58481

def event58483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 58467

def event58484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 58483 .coefficient))

def event58485 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event58486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26286⟩⟩) 0 ⟨11173⟩ 58485

def event58487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26286⟩⟩) (.authority (.programFamilyFact))

def exact58488RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩, (1)⟩]

theorem exact58488RawTermsValid :
    exact58488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26286⟩⟩) exact58488RawTerms (.finite 30) 58487 .exactZero (none)

def event58489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13101⟩⟩) 0 ⟨11173⟩ 58485

def event58490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13101⟩⟩) (.authority (.programFamilyFact))

def exact58491RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩], []⟩, (1)⟩]

theorem exact58491RawTermsValid :
    exact58491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58491 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13101⟩⟩) exact58491RawTerms (.finite 30) 58490 .exactZero (none)

def event58492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26287⟩⟩) 0 ⟨13101⟩ 58491

def event58493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26287⟩⟩) 1 ⟨26286⟩ 58488

def event58494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26287⟩⟩) (.product (.predecessor 0 58492 .coefficient) (.predecessor 1 58493 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event58495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26287⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩) [⟨.result 58491 .coefficient, true, some 1⟩, ⟨.result 58488 .coefficient, true, some 1⟩])

def event58496 : Event := .survivorFold (1) 58495

def exact58497RawTerms : List Term := []

theorem exact58497RawTermsValid :
    exact58497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58497 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26287⟩⟩) exact58497RawTerms (.finite 900) 58494 (.finite 900) (some (58495))

def event58498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26288⟩⟩) 0 ⟨26287⟩ 58497

def event58499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26288⟩⟩) (.identity (.predecessor 0 58498 .coefficient))

def event58500 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26288⟩⟩) (.finite 900)

def event58501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26472⟩⟩) 0 ⟨26288⟩ 58500

def event58502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26472⟩⟩) (.authority (.programFamilyFact))

def exact58503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], []⟩, (1)⟩]

theorem exact58503RawTermsValid :
    exact58503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26472⟩⟩) exact58503RawTerms (.finite 30) 58502 .exactZero (none)

def event58504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26473⟩⟩) 0 ⟨26472⟩ 58503

def event58505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26473⟩⟩) (.identity (.predecessor 0 58504 .coefficient))

def event58506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26473⟩⟩) (.finite 30)

def event58507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27312⟩⟩) 0 ⟨26473⟩ 58506

def event58508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27312⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact58509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩]

theorem exact58509RawTermsValid :
    exact58509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58509 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27312⟩⟩) exact58509RawTerms (.finite 5647228698) 58508 .exactZero (none)

def event58510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact58511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact58511RawTermsValid :
    exact58511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact58511RawTerms .large 58510 .exactZero (none)

def event58512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27313⟩⟩) 0 ⟨35⟩ 58511

def event58513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27313⟩⟩) 1 ⟨27312⟩ 58509

def event58514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27313⟩⟩) (.product (.predecessor 0 58512 .coefficient) (.predecessor 1 58513 .coefficient) (⟨false, false, none, none, none⟩))

def event58515 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27313⟩⟩, .operator (⟨58511, 0⟩, ⟨58509, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩)

def exact58516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩]

theorem exact58516RawTermsValid :
    exact58516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27313⟩⟩) exact58516RawTerms .large 58514 .exactZero (none)

def event58517 : Event := .preFoldPolynomial 58516 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩] .exactZero none

def exact58518RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27312⟩⟩]⟩, (1)⟩]

def event58518 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27313⟩⟩) 58517 exact58518RawTerms .large 58514 .exactZero (none)

def event58519 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28488⟩⟩)

def event58520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event58521 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event58522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event58523 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event58524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event58525 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event58526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event58527 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event58528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 58527

def event58529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 58525

def event58530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 58528 .coefficient) (.value (.predecessor 1 58529 .coefficient)))

def event58531 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event58532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 58531

def event58533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 58523

def event58534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 58532 .coefficient, .predecessor 1 58533 .coefficient])

def event58535 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event58536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 58535

def event58537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 58521

def event58538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 58537 .coefficient))

def event58539 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event58540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26286⟩⟩) 0 ⟨11173⟩ 58539

def event58541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26286⟩⟩) (.authority (.programFamilyFact))

def exact58542RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩, (1)⟩]

theorem exact58542RawTermsValid :
    exact58542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58542 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26286⟩⟩) exact58542RawTerms (.finite 30) 58541 .exactZero (none)

def event58543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13101⟩⟩) 0 ⟨11173⟩ 58539

def event58544 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13101⟩⟩) (.authority (.programFamilyFact))

def exact58545RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩], []⟩, (1)⟩]

theorem exact58545RawTermsValid :
    exact58545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13101⟩⟩) exact58545RawTerms (.finite 30) 58544 .exactZero (none)

def event58546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26287⟩⟩) 0 ⟨13101⟩ 58545

def event58547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26287⟩⟩) 1 ⟨26286⟩ 58542

def event58548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26287⟩⟩) (.product (.predecessor 0 58546 .coefficient) (.predecessor 1 58547 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event58549 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26287⟩⟩, .operator (⟨58545, 0⟩, ⟨58542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩, (1)⟩)

def exact58550RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13101⟩⟩, ⟨.program ⟨257⟩, ⟨26286⟩⟩], []⟩, (1)⟩]

theorem exact58550RawTermsValid :
    exact58550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26287⟩⟩) exact58550RawTerms (.finite 900) 58548 .exactZero (none)

def event58551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26288⟩⟩) 0 ⟨26287⟩ 58550

def event58552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26288⟩⟩) (.identity (.predecessor 0 58551 .coefficient))

def event58553 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26288⟩⟩) (.finite 900)

def event58554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26472⟩⟩) 0 ⟨26288⟩ 58553

def event58555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26472⟩⟩) (.authority (.programFamilyFact))

def exact58556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], []⟩, (1)⟩]

theorem exact58556RawTermsValid :
    exact58556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26472⟩⟩) exact58556RawTerms (.finite 30) 58555 .exactZero (none)

def event58557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26473⟩⟩) 0 ⟨26472⟩ 58556

def event58558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26473⟩⟩) (.identity (.predecessor 0 58557 .coefficient))

def event58559 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26473⟩⟩) (.finite 30)

def event58560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27631⟩⟩) 0 ⟨26473⟩ 58559

def event58561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27631⟩⟩) (.authority (.programFamilyFact))

def event58562 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27631⟩⟩) (.finite 3720)

def event58563 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event58564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27632⟩⟩) 0 ⟨7177⟩ 58563

def event58565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27632⟩⟩) 1 ⟨27631⟩ 58562

def event58566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27632⟩⟩) (.authority (.operator))

def exact58567RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (1)⟩]

theorem exact58567RawTermsValid :
    exact58567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58567 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27632⟩⟩) exact58567RawTerms .large 58566 .exactZero (none)

def event58568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28483⟩⟩) 0 ⟨27632⟩ 58567

def event58569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28483⟩⟩) (.authority (.operator))

def exact58570RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩]

theorem exact58570RawTermsValid :
    exact58570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28483⟩⟩) exact58570RawTerms (.finite 8192) 58569 .exactZero (none)

def event58571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event58572 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event58573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27798⟩⟩) 0 ⟨26473⟩ 58559

def event58574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27798⟩⟩) 1 ⟨136⟩ 58572

def event58575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27798⟩⟩) (.sum [.predecessor 0 58573 .coefficient, .predecessor 1 58574 .coefficient])

def event58576 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27798⟩⟩) (.finite 30)

def event58577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27799⟩⟩) 0 ⟨27798⟩ 58576

def event58578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27799⟩⟩) (.identity (.predecessor 0 58577 .coefficient))

def exact58579RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], []⟩, (1)⟩]

theorem exact58579RawTermsValid :
    exact58579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27799⟩⟩) exact58579RawTerms (.finite 30) 58578 .exactZero (none)

def event58580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact58581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58581RawTermsValid :
    exact58581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact58581RawTerms .large 58580 .exactZero (none)

def event58582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27800⟩⟩) 0 ⟨6908⟩ 58581

def event58583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27800⟩⟩) 1 ⟨27799⟩ 58579

def event58584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27800⟩⟩) (.product (.predecessor 0 58582 .coefficient) (.predecessor 1 58583 .coefficient) (⟨false, false, none, none, none⟩))

def event58585 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27800⟩⟩, .operator (⟨58581, 0⟩, ⟨58579, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact58586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58586RawTermsValid :
    exact58586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27800⟩⟩) exact58586RawTerms .large 58584 .exactZero (none)

def event58587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 58563

def event58588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact58589RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact58589RawTermsValid :
    exact58589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact58589RawTerms .large 58588 .exactZero (none)

def event58590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27801⟩⟩) 0 ⟨7189⟩ 58589

def event58591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27801⟩⟩) 1 ⟨27800⟩ 58586

def event58592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27801⟩⟩) (.sum [.predecessor 0 58590 .coefficient, .predecessor 1 58591 .coefficient])

def exact58593RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58593RawTermsValid :
    exact58593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27801⟩⟩) exact58593RawTerms .large 58592 .exactZero (none)

def event58594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28484⟩⟩) 0 ⟨27801⟩ 58593

def event58595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28484⟩⟩) 1 ⟨28483⟩ 58570

def event58596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28484⟩⟩) (.product (.predecessor 0 58594 .coefficient) (.predecessor 1 58595 .coefficient) (⟨false, false, none, none, none⟩))

def event58597 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28484⟩⟩, .operator (⟨58593, 0⟩, ⟨58570, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩)

def event58598 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28484⟩⟩, .operator (⟨58593, 1⟩, ⟨58570, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (-1)⟩)

def event58599 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28484⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28483⟩⟩) ⟨27632⟩ 58567)

def event58600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28484⟩⟩, .relation 58599 0, ⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (-1)⟩)

def exact58601RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (-1)⟩]

theorem exact58601RawTermsValid :
    exact58601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28484⟩⟩) exact58601RawTerms .large 58596 .exactZero (none)

def event58602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26726⟩⟩) 0 ⟨26473⟩ 58559

def event58603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26726⟩⟩) (.authority (.programFamilyFact))

def exact58604RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], []⟩, (1)⟩]

theorem exact58604RawTermsValid :
    exact58604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26726⟩⟩) exact58604RawTerms (.finite 30) 58603 .exactZero (none)

def event58605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26728⟩⟩) 0 ⟨6908⟩ 58581

def event58606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26728⟩⟩) 1 ⟨26726⟩ 58604

def event58607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26728⟩⟩) (.product (.predecessor 0 58605 .coefficient) (.predecessor 1 58606 .coefficient) (⟨false, true, none, none, some 1⟩))

def event58608 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26728⟩⟩, .operator (⟨58581, 0⟩, ⟨58604, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact58609RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact58609RawTermsValid :
    exact58609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58609 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26728⟩⟩) exact58609RawTerms .large 58607 .exactZero (none)

def event58610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7217⟩⟩) 0 ⟨7177⟩ 58563

def event58611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7217⟩⟩) (.authority (.operator))

def exact58612RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩]

theorem exact58612RawTermsValid :
    exact58612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58612 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7217⟩⟩) exact58612RawTerms .large 58611 .exactZero (none)

def event58613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26729⟩⟩) 0 ⟨7217⟩ 58612

def event58614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26729⟩⟩) 1 ⟨26728⟩ 58609

def event58615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26729⟩⟩) (.sum [.predecessor 0 58613 .coefficient, .predecessor 1 58614 .coefficient])

def exact58616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58616RawTermsValid :
    exact58616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26729⟩⟩) exact58616RawTerms .large 58615 .exactZero (none)

def event58617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28488⟩⟩) 0 ⟨26729⟩ 58616

def event58618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28488⟩⟩) 1 ⟨28484⟩ 58601

def event58619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28488⟩⟩) (.sum [.predecessor 0 58617 .coefficient, .predecessor 1 58618 .coefficient])

def exact58620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact58620RawTermsValid :
    exact58620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event58620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28488⟩⟩) exact58620RawTerms .large 58619 .exactZero (none)

def event58621 : Event := .preFoldPolynomial 58620 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact58622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28483⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26472⟩⟩], [⟨.program ⟨257⟩, ⟨27632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event58622 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28488⟩⟩) 58621 exact58622RawTerms .large 58619 .exactZero (none)

def event58623 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26473⟩⟩) ⟨⟨96⟩, ⟨78⟩, ⟨135⟩⟩ ⟨58465, 58623⟩

def eventLeaf3648 : Array AnnotatedEvent := #[
  { event := event58368
    frameStart := 58307 },
  { event := event58369
    frameStart := 58307 },
  { event := event58370
    frameStart := 58307 },
  { event := event58371
    frameStart := 58307 },
  { event := event58372
    frameStart := 58307 },
  { event := event58373
    frameStart := 58307 },
  { event := event58374
    frameStart := 58307 },
  { event := event58375
    frameStart := 58307 },
  { event := event58376
    frameStart := 58307 },
  { event := event58377
    frameStart := 58307 },
  { event := event58378
    frameStart := 58307 },
  { event := event58379
    frameStart := 58307 },
  { event := event58380
    frameStart := 58307 },
  { event := event58381
    frameStart := 58307 },
  { event := event58382
    frameStart := 58307 },
  { event := event58383
    frameStart := 58307 }
]

def eventLeaf3649 : Array AnnotatedEvent := #[
  { event := event58384
    frameStart := 58307 },
  { event := event58385
    frameStart := 58307 },
  { event := event58386
    frameStart := 58307 },
  { event := event58387
    frameStart := 58307 },
  { event := event58388
    frameStart := 58307 },
  { event := event58389
    frameStart := 58307 },
  { event := event58390
    frameStart := 58307 },
  { event := event58391
    frameStart := 58307 },
  { event := event58392
    frameStart := 58307 },
  { event := event58393
    frameStart := 58307 },
  { event := event58394
    frameStart := 58307 },
  { event := event58395
    frameStart := 58307 },
  { event := event58396
    frameStart := 58307 },
  { event := event58397
    frameStart := 58307 },
  { event := event58398
    frameStart := 58307 },
  { event := event58399
    frameStart := 58307 }
]

def eventLeaf3650 : Array AnnotatedEvent := #[
  { event := event58400
    frameStart := 58307 },
  { event := event58401
    frameStart := 58307 },
  { event := event58402
    frameStart := 58307 },
  { event := event58403
    frameStart := 58307 },
  { event := event58404
    frameStart := 58307 },
  { event := event58405
    frameStart := 58307 },
  { event := event58406
    frameStart := 58307 },
  { event := event58407
    frameStart := 58307 },
  { event := event58408
    frameStart := 58307 },
  { event := event58409
    frameStart := 58307 },
  { event := event58410
    frameStart := 58307 },
  { event := event58411
    frameStart := 0 },
  { event := event58412
    frameStart := 0 },
  { event := event58413
    frameStart := 0 },
  { event := event58414
    frameStart := 0 },
  { event := event58415
    frameStart := 0 }
]

def eventLeaf3651 : Array AnnotatedEvent := #[
  { event := event58416
    frameStart := 0 },
  { event := event58417
    frameStart := 0 },
  { event := event58418
    frameStart := 0 },
  { event := event58419
    frameStart := 0 },
  { event := event58420
    frameStart := 0 },
  { event := event58421
    frameStart := 0 },
  { event := event58422
    frameStart := 0 },
  { event := event58423
    frameStart := 0 },
  { event := event58424
    frameStart := 0 },
  { event := event58425
    frameStart := 0 },
  { event := event58426
    frameStart := 0 },
  { event := event58427
    frameStart := 0 },
  { event := event58428
    frameStart := 0 },
  { event := event58429
    frameStart := 0 },
  { event := event58430
    frameStart := 0 },
  { event := event58431
    frameStart := 0 }
]

def eventLeaf3652 : Array AnnotatedEvent := #[
  { event := event58432
    frameStart := 0 },
  { event := event58433
    frameStart := 0 },
  { event := event58434
    frameStart := 0 },
  { event := event58435
    frameStart := 0 },
  { event := event58436
    frameStart := 0 },
  { event := event58437
    frameStart := 0 },
  { event := event58438
    frameStart := 0 },
  { event := event58439
    frameStart := 0 },
  { event := event58440
    frameStart := 0 },
  { event := event58441
    frameStart := 0 },
  { event := event58442
    frameStart := 0 },
  { event := event58443
    frameStart := 0 },
  { event := event58444
    frameStart := 0 },
  { event := event58445
    frameStart := 0 },
  { event := event58446
    frameStart := 0 },
  { event := event58447
    frameStart := 0 }
]

def eventLeaf3653 : Array AnnotatedEvent := #[
  { event := event58448
    frameStart := 0 },
  { event := event58449
    frameStart := 0 },
  { event := event58450
    frameStart := 0 },
  { event := event58451
    frameStart := 0 },
  { event := event58452
    frameStart := 0 },
  { event := event58453
    frameStart := 0 },
  { event := event58454
    frameStart := 0 },
  { event := event58455
    frameStart := 0 },
  { event := event58456
    frameStart := 0 },
  { event := event58457
    frameStart := 0 },
  { event := event58458
    frameStart := 0 },
  { event := event58459
    frameStart := 0 },
  { event := event58460
    frameStart := 0 },
  { event := event58461
    frameStart := 0 },
  { event := event58462
    frameStart := 0 },
  { event := event58463
    frameStart := 0 }
]

def eventLeaf3654 : Array AnnotatedEvent := #[
  { event := event58464
    frameStart := 0 },
  { event := event58465
    frameStart := 58465 },
  { event := event58466
    frameStart := 58465 },
  { event := event58467
    frameStart := 58465 },
  { event := event58468
    frameStart := 58465 },
  { event := event58469
    frameStart := 58465 },
  { event := event58470
    frameStart := 58465 },
  { event := event58471
    frameStart := 58465 },
  { event := event58472
    frameStart := 58465 },
  { event := event58473
    frameStart := 58465 },
  { event := event58474
    frameStart := 58465 },
  { event := event58475
    frameStart := 58465 },
  { event := event58476
    frameStart := 58465 },
  { event := event58477
    frameStart := 58465 },
  { event := event58478
    frameStart := 58465 },
  { event := event58479
    frameStart := 58465 }
]

def eventLeaf3655 : Array AnnotatedEvent := #[
  { event := event58480
    frameStart := 58465 },
  { event := event58481
    frameStart := 58465 },
  { event := event58482
    frameStart := 58465 },
  { event := event58483
    frameStart := 58465 },
  { event := event58484
    frameStart := 58465 },
  { event := event58485
    frameStart := 58465 },
  { event := event58486
    frameStart := 58465 },
  { event := event58487
    frameStart := 58465 },
  { event := event58488
    frameStart := 58465 },
  { event := event58489
    frameStart := 58465 },
  { event := event58490
    frameStart := 58465 },
  { event := event58491
    frameStart := 58465 },
  { event := event58492
    frameStart := 58465 },
  { event := event58493
    frameStart := 58465 },
  { event := event58494
    frameStart := 58465 },
  { event := event58495
    frameStart := 58465 }
]

def eventLeaf3656 : Array AnnotatedEvent := #[
  { event := event58496
    frameStart := 58465 },
  { event := event58497
    frameStart := 58465 },
  { event := event58498
    frameStart := 58465 },
  { event := event58499
    frameStart := 58465 },
  { event := event58500
    frameStart := 58465 },
  { event := event58501
    frameStart := 58465 },
  { event := event58502
    frameStart := 58465 },
  { event := event58503
    frameStart := 58465 },
  { event := event58504
    frameStart := 58465 },
  { event := event58505
    frameStart := 58465 },
  { event := event58506
    frameStart := 58465 },
  { event := event58507
    frameStart := 58465 },
  { event := event58508
    frameStart := 58465 },
  { event := event58509
    frameStart := 58465 },
  { event := event58510
    frameStart := 58465 },
  { event := event58511
    frameStart := 58465 }
]

def eventLeaf3657 : Array AnnotatedEvent := #[
  { event := event58512
    frameStart := 58465 },
  { event := event58513
    frameStart := 58465 },
  { event := event58514
    frameStart := 58465 },
  { event := event58515
    frameStart := 58465 },
  { event := event58516
    frameStart := 58465 },
  { event := event58517
    frameStart := 58465 },
  { event := event58518
    frameStart := 58465 },
  { event := event58519
    frameStart := 58519 },
  { event := event58520
    frameStart := 58519 },
  { event := event58521
    frameStart := 58519 },
  { event := event58522
    frameStart := 58519 },
  { event := event58523
    frameStart := 58519 },
  { event := event58524
    frameStart := 58519 },
  { event := event58525
    frameStart := 58519 },
  { event := event58526
    frameStart := 58519 },
  { event := event58527
    frameStart := 58519 }
]

def eventLeaf3658 : Array AnnotatedEvent := #[
  { event := event58528
    frameStart := 58519 },
  { event := event58529
    frameStart := 58519 },
  { event := event58530
    frameStart := 58519 },
  { event := event58531
    frameStart := 58519 },
  { event := event58532
    frameStart := 58519 },
  { event := event58533
    frameStart := 58519 },
  { event := event58534
    frameStart := 58519 },
  { event := event58535
    frameStart := 58519 },
  { event := event58536
    frameStart := 58519 },
  { event := event58537
    frameStart := 58519 },
  { event := event58538
    frameStart := 58519 },
  { event := event58539
    frameStart := 58519 },
  { event := event58540
    frameStart := 58519 },
  { event := event58541
    frameStart := 58519 },
  { event := event58542
    frameStart := 58519 },
  { event := event58543
    frameStart := 58519 }
]

def eventLeaf3659 : Array AnnotatedEvent := #[
  { event := event58544
    frameStart := 58519 },
  { event := event58545
    frameStart := 58519 },
  { event := event58546
    frameStart := 58519 },
  { event := event58547
    frameStart := 58519 },
  { event := event58548
    frameStart := 58519 },
  { event := event58549
    frameStart := 58519 },
  { event := event58550
    frameStart := 58519 },
  { event := event58551
    frameStart := 58519 },
  { event := event58552
    frameStart := 58519 },
  { event := event58553
    frameStart := 58519 },
  { event := event58554
    frameStart := 58519 },
  { event := event58555
    frameStart := 58519 },
  { event := event58556
    frameStart := 58519 },
  { event := event58557
    frameStart := 58519 },
  { event := event58558
    frameStart := 58519 },
  { event := event58559
    frameStart := 58519 }
]

def eventLeaf3660 : Array AnnotatedEvent := #[
  { event := event58560
    frameStart := 58519 },
  { event := event58561
    frameStart := 58519 },
  { event := event58562
    frameStart := 58519 },
  { event := event58563
    frameStart := 58519 },
  { event := event58564
    frameStart := 58519 },
  { event := event58565
    frameStart := 58519 },
  { event := event58566
    frameStart := 58519 },
  { event := event58567
    frameStart := 58519 },
  { event := event58568
    frameStart := 58519 },
  { event := event58569
    frameStart := 58519 },
  { event := event58570
    frameStart := 58519 },
  { event := event58571
    frameStart := 58519 },
  { event := event58572
    frameStart := 58519 },
  { event := event58573
    frameStart := 58519 },
  { event := event58574
    frameStart := 58519 },
  { event := event58575
    frameStart := 58519 }
]

def eventLeaf3661 : Array AnnotatedEvent := #[
  { event := event58576
    frameStart := 58519 },
  { event := event58577
    frameStart := 58519 },
  { event := event58578
    frameStart := 58519 },
  { event := event58579
    frameStart := 58519 },
  { event := event58580
    frameStart := 58519 },
  { event := event58581
    frameStart := 58519 },
  { event := event58582
    frameStart := 58519 },
  { event := event58583
    frameStart := 58519 },
  { event := event58584
    frameStart := 58519 },
  { event := event58585
    frameStart := 58519 },
  { event := event58586
    frameStart := 58519 },
  { event := event58587
    frameStart := 58519 },
  { event := event58588
    frameStart := 58519 },
  { event := event58589
    frameStart := 58519 },
  { event := event58590
    frameStart := 58519 },
  { event := event58591
    frameStart := 58519 }
]

def eventLeaf3662 : Array AnnotatedEvent := #[
  { event := event58592
    frameStart := 58519 },
  { event := event58593
    frameStart := 58519 },
  { event := event58594
    frameStart := 58519 },
  { event := event58595
    frameStart := 58519 },
  { event := event58596
    frameStart := 58519 },
  { event := event58597
    frameStart := 58519 },
  { event := event58598
    frameStart := 58519 },
  { event := event58599
    frameStart := 58519 },
  { event := event58600
    frameStart := 58519 },
  { event := event58601
    frameStart := 58519 },
  { event := event58602
    frameStart := 58519 },
  { event := event58603
    frameStart := 58519 },
  { event := event58604
    frameStart := 58519 },
  { event := event58605
    frameStart := 58519 },
  { event := event58606
    frameStart := 58519 },
  { event := event58607
    frameStart := 58519 }
]

def eventLeaf3663 : Array AnnotatedEvent := #[
  { event := event58608
    frameStart := 58519 },
  { event := event58609
    frameStart := 58519 },
  { event := event58610
    frameStart := 58519 },
  { event := event58611
    frameStart := 58519 },
  { event := event58612
    frameStart := 58519 },
  { event := event58613
    frameStart := 58519 },
  { event := event58614
    frameStart := 58519 },
  { event := event58615
    frameStart := 58519 },
  { event := event58616
    frameStart := 58519 },
  { event := event58617
    frameStart := 58519 },
  { event := event58618
    frameStart := 58519 },
  { event := event58619
    frameStart := 58519 },
  { event := event58620
    frameStart := 58519 },
  { event := event58621
    frameStart := 58519 },
  { event := event58622
    frameStart := 58519 },
  { event := event58623
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events228
