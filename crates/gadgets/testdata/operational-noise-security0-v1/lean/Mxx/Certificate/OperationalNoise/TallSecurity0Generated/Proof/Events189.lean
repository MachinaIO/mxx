import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events189

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event48384 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28102⟩⟩) 0 ⟨24230⟩ 48383

def event48385 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28102⟩⟩) (.authority (.operator))

def exact48386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩]

theorem exact48386RawTermsValid :
    exact48386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48386 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28102⟩⟩) exact48386RawTerms (.finite 8192) 48385 .exactZero (none)

def event48387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event48388 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event48389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16142⟩⟩) 0 ⟨16068⟩ 48375

def event48390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16142⟩⟩) 1 ⟨110⟩ 48388

def event48391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16142⟩⟩) (.sum [.predecessor 0 48389 .coefficient, .predecessor 1 48390 .coefficient])

def event48392 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16142⟩⟩) (.finite 22)

def event48393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16143⟩⟩) 0 ⟨16142⟩ 48392

def event48394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16143⟩⟩) (.identity (.predecessor 0 48393 .coefficient))

def exact48395RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], []⟩, (1)⟩]

theorem exact48395RawTermsValid :
    exact48395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16143⟩⟩) exact48395RawTerms (.finite 22) 48394 .exactZero (none)

def event48396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact48397RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48397RawTermsValid :
    exact48397RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48397 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact48397RawTerms .large 48396 .exactZero (none)

def event48398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16144⟩⟩) 0 ⟨6544⟩ 48397

def event48399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16144⟩⟩) 1 ⟨16143⟩ 48395

def event48400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16144⟩⟩) (.product (.predecessor 0 48398 .coefficient) (.predecessor 1 48399 .coefficient) (⟨false, false, none, none, none⟩))

def event48401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16144⟩⟩, .operator (⟨48397, 0⟩, ⟨48395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48402RawTermsValid :
    exact48402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16144⟩⟩) exact48402RawTerms .large 48400 .exactZero (none)

def event48403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 48379

def event48404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact48405RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact48405RawTermsValid :
    exact48405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact48405RawTerms .large 48404 .exactZero (none)

def event48406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16145⟩⟩) 0 ⟨6698⟩ 48405

def event48407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16145⟩⟩) 1 ⟨16144⟩ 48402

def event48408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16145⟩⟩) (.sum [.predecessor 0 48406 .coefficient, .predecessor 1 48407 .coefficient])

def exact48409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48409RawTermsValid :
    exact48409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16145⟩⟩) exact48409RawTerms .large 48408 .exactZero (none)

def event48410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28103⟩⟩) 0 ⟨16145⟩ 48409

def event48411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28103⟩⟩) 1 ⟨28102⟩ 48386

def event48412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28103⟩⟩) (.product (.predecessor 0 48410 .coefficient) (.predecessor 1 48411 .coefficient) (⟨false, false, none, none, none⟩))

def event48413 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28103⟩⟩, .operator (⟨48409, 0⟩, ⟨48386, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩)

def event48414 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28103⟩⟩, .operator (⟨48409, 1⟩, ⟨48386, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩)

def event48415 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28103⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28102⟩⟩) ⟨24230⟩ 48383)

def event48416 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28103⟩⟩, .relation 48415 0, ⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (-1)⟩)

def exact48417RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (-1)⟩]

theorem exact48417RawTermsValid :
    exact48417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48417 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28103⟩⟩) exact48417RawTerms .large 48412 .exactZero (none)

def event48418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18049⟩⟩) 0 ⟨16068⟩ 48375

def event48419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18049⟩⟩) (.authority (.programFamilyFact))

def exact48420RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩, (1)⟩]

theorem exact48420RawTermsValid :
    exact48420RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48420 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18049⟩⟩) exact48420RawTerms (.finite 22) 48419 .exactZero (none)

def event48421 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18054⟩⟩) 0 ⟨6544⟩ 48397

def event48422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18054⟩⟩) 1 ⟨18049⟩ 48420

def event48423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18054⟩⟩) (.product (.predecessor 0 48421 .coefficient) (.predecessor 1 48422 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48424 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18054⟩⟩, .operator (⟨48397, 0⟩, ⟨48420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48425RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48425RawTermsValid :
    exact48425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18054⟩⟩) exact48425RawTerms .large 48423 .exactZero (none)

def event48426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6724⟩⟩) 0 ⟨6689⟩ 48379

def event48427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6724⟩⟩) (.authority (.operator))

def exact48428RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩]

theorem exact48428RawTermsValid :
    exact48428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48428 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6724⟩⟩) exact48428RawTerms .large 48427 .exactZero (none)

def event48429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18055⟩⟩) 0 ⟨6724⟩ 48428

def event48430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18055⟩⟩) 1 ⟨18054⟩ 48425

def event48431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18055⟩⟩) (.sum [.predecessor 0 48429 .coefficient, .predecessor 1 48430 .coefficient])

def exact48432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48432RawTermsValid :
    exact48432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48432 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18055⟩⟩) exact48432RawTerms .large 48431 .exactZero (none)

def event48433 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28108⟩⟩) 0 ⟨18055⟩ 48432

def event48434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28108⟩⟩) 1 ⟨28103⟩ 48417

def event48435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28108⟩⟩) (.sum [.predecessor 0 48433 .coefficient, .predecessor 1 48434 .coefficient])

def exact48436RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48436RawTermsValid :
    exact48436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48436 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28108⟩⟩) exact48436RawTerms .large 48435 .exactZero (none)

def event48437 : Event := .preFoldPolynomial 48436 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event48438 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28108⟩⟩) 48437 exact48438RawTerms .large 48435 .exactZero (none)

def event48439 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16068⟩⟩) ⟨⟨137⟩, ⟨45⟩, ⟨109⟩⟩ ⟨48281, 48439⟩

def event48440 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21483⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩) (1) 0 2 (.universal 48439 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩) (none) 48438)

def event48441 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21483⟩⟩, .relation 48440 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩)

def event48442 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21483⟩⟩, .relation 48440 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩)

def event48443 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21483⟩⟩, .relation 48440 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩)

def event48444 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21483⟩⟩, .relation 48440 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48445RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48445RawTermsValid :
    exact48445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21483⟩⟩) exact48445RawTerms .large 48277 (.finite 1811303510016) (some (48279))

def event48446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28105⟩⟩) 0 ⟨21483⟩ 48445

def event48447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28105⟩⟩) 1 ⟨28104⟩ 48267

def event48448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28105⟩⟩) (.sum [.predecessor 0 48446 .coefficient, .predecessor 1 48447 .coefficient])

def event48449 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28105⟩⟩, .operator (⟨48445, 0⟩, ⟨48267, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩, (1)⟩)

def event48450 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28105⟩⟩, .operator (⟨48445, 2⟩, ⟨48267, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], [⟨.program ⟨214⟩, ⟨24230⟩⟩]⟩, (-1)⟩)

def event48451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28105⟩⟩) (.sum [.result 48445 .summary, .result 48267 .summary])

def exact48452RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48452RawTermsValid :
    exact48452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48452 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28105⟩⟩) exact48452RawTerms .large 48448 (.finite 1292113298829627502592) (some (48451))

def event48453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28106⟩⟩) 0 ⟨28105⟩ 48452

def event48454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28106⟩⟩) 1 ⟨6638⟩ 5699

def event48455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28106⟩⟩) (.product (.predecessor 0 48453 .coefficient) (.predecessor 1 48454 .coefficient) (⟨false, false, none, none, none⟩))

def event48456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28106⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) [⟨.result 5695 .coefficient, false, none⟩])

def event48457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28106⟩⟩) (.product (.result 48452 .summary) (.transfer 48456) (⟨false, false, none, none, none⟩))

def event48458 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28106⟩⟩, .operator (⟨48452, 0⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩)

def event48459 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28106⟩⟩, .operator (⟨48452, 1⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (-1)⟩)

def event48460 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28106⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6637⟩⟩) ⟨6590⟩ 5692)

def event48461 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28106⟩⟩, .relation 48460 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48462RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48462RawTermsValid :
    exact48462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28106⟩⟩) exact48462RawTerms .large 48455 (.finite 4742076480517514208552681472) (some (48457))

def event48463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24167⟩⟩) 0 ⟨6689⟩ 5477

def event48464 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24167⟩⟩) 1 ⟨24166⟩ 40859

def event48465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24167⟩⟩) (.authority (.operator))

def exact48466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩]

theorem exact48466RawTermsValid :
    exact48466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48466 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24167⟩⟩) exact48466RawTerms .large 48465 .exactZero (none)

def event48467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27885⟩⟩) 0 ⟨24167⟩ 48466

def event48468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27885⟩⟩) (.authority (.operator))

def exact48469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩]

theorem exact48469RawTermsValid :
    exact48469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27885⟩⟩) exact48469RawTerms (.finite 8192) 48468 .exactZero (none)

def event48470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27887⟩⟩) 0 ⟨26078⟩ 41143

def event48471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27887⟩⟩) 1 ⟨27885⟩ 48469

def event48472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27887⟩⟩) (.product (.predecessor 0 48470 .coefficient) (.predecessor 1 48471 .coefficient) (⟨false, false, none, none, none⟩))

def event48473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27887⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩) [⟨.result 48469 .coefficient, false, none⟩])

def event48474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27887⟩⟩) (.product (.result 41143 .summary) (.transfer 48473) (⟨false, false, none, none, none⟩))

def event48475 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27887⟩⟩, .operator (⟨41143, 0⟩, ⟨48469, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩)

def event48476 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27887⟩⟩, .operator (⟨41143, 1⟩, ⟨48469, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩)

def event48477 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27887⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27885⟩⟩) ⟨24167⟩ 48466)

def event48478 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27887⟩⟩, .relation 48477 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (-1)⟩)

def exact48479RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (-1)⟩]

theorem exact48479RawTermsValid :
    exact48479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27887⟩⟩) exact48479RawTerms .large 48472 (.finite 1292068472128282820608) (some (48474))

def event48480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21336⟩⟩) 0 ⟨15949⟩ 1837

def event48481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21336⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact48482RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩]

theorem exact48482RawTermsValid :
    exact48482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48482 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21336⟩⟩) exact48482RawTerms (.finite 136065468) 48481 .exactZero (none)

def event48483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21338⟩⟩) 0 ⟨21336⟩ 48482

def event48484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21338⟩⟩) 1 ⟨2348⟩ 4

def event48485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21338⟩⟩) (.scale (.predecessor 0 48483 .coefficient) (.value (.predecessor 1 48484 .coefficient)))

def exact48486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩]

theorem exact48486RawTermsValid :
    exact48486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48486 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21338⟩⟩) exact48486RawTerms (.finite 136065468) 48485 .exactZero (none)

def event48487 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21339⟩⟩) 0 ⟨5553⟩ 36137

def event48488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21339⟩⟩) 1 ⟨21338⟩ 48486

def event48489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21339⟩⟩) (.product (.predecessor 0 48487 .coefficient) (.predecessor 1 48488 .coefficient) (⟨false, false, none, none, none⟩))

def event48490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21339⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩) [⟨.result 48482 .coefficient, false, none⟩])

def event48491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21339⟩⟩) (.product (.result 36137 .summary) (.transfer 48490) (⟨false, false, none, none, none⟩))

def event48492 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21339⟩⟩, .operator (⟨36137, 0⟩, ⟨48486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩)

def event48493 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21337⟩⟩)

def event48494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48495 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48497 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48499 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48501 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48501

def event48503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48499

def event48504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48502 .coefficient) (.value (.predecessor 1 48503 .coefficient)))

def event48505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48505

def event48507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48497

def event48508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48506 .coefficient, .predecessor 1 48507 .coefficient])

def event48509 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48509

def event48511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48495

def event48512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48511 .coefficient))

def event48513 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11477⟩⟩) 0 ⟨5548⟩ 48513

def event48515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11477⟩⟩) (.authority (.programFamilyFact))

def exact48516RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩], []⟩, (1)⟩]

theorem exact48516RawTermsValid :
    exact48516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48516 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11477⟩⟩) exact48516RawTerms (.finite 18) 48515 .exactZero (none)

def event48517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14225⟩⟩) 0 ⟨5548⟩ 48513

def event48518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14225⟩⟩) (.authority (.programFamilyFact))

def exact48519RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩, (1)⟩]

theorem exact48519RawTermsValid :
    exact48519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48519 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14225⟩⟩) exact48519RawTerms (.finite 18) 48518 .exactZero (none)

def event48520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14226⟩⟩) 0 ⟨14225⟩ 48519

def event48521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14226⟩⟩) 1 ⟨11477⟩ 48516

def event48522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14226⟩⟩) (.product (.predecessor 0 48520 .coefficient) (.predecessor 1 48521 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14226⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩) [⟨.result 48519 .coefficient, true, some 1⟩, ⟨.result 48516 .coefficient, true, some 1⟩])

def event48524 : Event := .survivorFold (1) 48523

def exact48525RawTerms : List Term := []

theorem exact48525RawTermsValid :
    exact48525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14226⟩⟩) exact48525RawTerms (.finite 324) 48522 (.finite 324) (some (48523))

def event48526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14227⟩⟩) 0 ⟨14226⟩ 48525

def event48527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14227⟩⟩) (.identity (.predecessor 0 48526 .coefficient))

def event48528 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14227⟩⟩) (.finite 324)

def event48529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15948⟩⟩) 0 ⟨14227⟩ 48528

def event48530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15948⟩⟩) (.authority (.programFamilyFact))

def exact48531RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], []⟩, (1)⟩]

theorem exact48531RawTermsValid :
    exact48531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15948⟩⟩) exact48531RawTerms (.finite 18) 48530 .exactZero (none)

def event48532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15949⟩⟩) 0 ⟨15948⟩ 48531

def event48533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15949⟩⟩) (.identity (.predecessor 0 48532 .coefficient))

def event48534 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15949⟩⟩) (.finite 18)

def event48535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21336⟩⟩) 0 ⟨15949⟩ 48534

def event48536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21336⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact48537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩]

theorem exact48537RawTermsValid :
    exact48537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21336⟩⟩) exact48537RawTerms (.finite 136065468) 48536 .exactZero (none)

def event48538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact48539RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact48539RawTermsValid :
    exact48539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48539 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact48539RawTerms .large 48538 .exactZero (none)

def event48540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21337⟩⟩) 0 ⟨6⟩ 48539

def event48541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21337⟩⟩) 1 ⟨21336⟩ 48537

def event48542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21337⟩⟩) (.product (.predecessor 0 48540 .coefficient) (.predecessor 1 48541 .coefficient) (⟨false, false, none, none, none⟩))

def event48543 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21337⟩⟩, .operator (⟨48539, 0⟩, ⟨48537, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩)

def exact48544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩]

theorem exact48544RawTermsValid :
    exact48544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21337⟩⟩) exact48544RawTerms .large 48542 .exactZero (none)

def event48545 : Event := .preFoldPolynomial 48544 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩] .exactZero none

def exact48546RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩, (1)⟩]

def event48546 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21337⟩⟩) 48545 exact48546RawTerms .large 48542 .exactZero (none)

def event48547 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27891⟩⟩)

def event48548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48549 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48553 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48555 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48555

def event48557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48553

def event48558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48556 .coefficient) (.value (.predecessor 1 48557 .coefficient)))

def event48559 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48559

def event48561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48551

def event48562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48560 .coefficient, .predecessor 1 48561 .coefficient])

def event48563 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48563

def event48565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48549

def event48566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48565 .coefficient))

def event48567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11477⟩⟩) 0 ⟨5548⟩ 48567

def event48569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11477⟩⟩) (.authority (.programFamilyFact))

def exact48570RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩], []⟩, (1)⟩]

theorem exact48570RawTermsValid :
    exact48570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48570 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11477⟩⟩) exact48570RawTerms (.finite 18) 48569 .exactZero (none)

def event48571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14225⟩⟩) 0 ⟨5548⟩ 48567

def event48572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14225⟩⟩) (.authority (.programFamilyFact))

def exact48573RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩, (1)⟩]

theorem exact48573RawTermsValid :
    exact48573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14225⟩⟩) exact48573RawTerms (.finite 18) 48572 .exactZero (none)

def event48574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14226⟩⟩) 0 ⟨14225⟩ 48573

def event48575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14226⟩⟩) 1 ⟨11477⟩ 48570

def event48576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14226⟩⟩) (.product (.predecessor 0 48574 .coefficient) (.predecessor 1 48575 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48577 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14226⟩⟩, .operator (⟨48573, 0⟩, ⟨48570, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩, (1)⟩)

def exact48578RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11477⟩⟩, ⟨.program ⟨214⟩, ⟨14225⟩⟩], []⟩, (1)⟩]

theorem exact48578RawTermsValid :
    exact48578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48578 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14226⟩⟩) exact48578RawTerms (.finite 324) 48576 .exactZero (none)

def event48579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14227⟩⟩) 0 ⟨14226⟩ 48578

def event48580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14227⟩⟩) (.identity (.predecessor 0 48579 .coefficient))

def event48581 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14227⟩⟩) (.finite 324)

def event48582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15948⟩⟩) 0 ⟨14227⟩ 48581

def event48583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15948⟩⟩) (.authority (.programFamilyFact))

def exact48584RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], []⟩, (1)⟩]

theorem exact48584RawTermsValid :
    exact48584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15948⟩⟩) exact48584RawTerms (.finite 18) 48583 .exactZero (none)

def event48585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15949⟩⟩) 0 ⟨15948⟩ 48584

def event48586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15949⟩⟩) (.identity (.predecessor 0 48585 .coefficient))

def event48587 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15949⟩⟩) (.finite 18)

def event48588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24166⟩⟩) 0 ⟨15949⟩ 48587

def event48589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24166⟩⟩) (.authority (.programFamilyFact))

def event48590 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24166⟩⟩) (.finite 3720)

def event48591 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event48592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24167⟩⟩) 0 ⟨6689⟩ 48591

def event48593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24167⟩⟩) 1 ⟨24166⟩ 48590

def event48594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24167⟩⟩) (.authority (.operator))

def exact48595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩]

theorem exact48595RawTermsValid :
    exact48595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48595 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24167⟩⟩) exact48595RawTerms .large 48594 .exactZero (none)

def event48596 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27885⟩⟩) 0 ⟨24167⟩ 48595

def event48597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27885⟩⟩) (.authority (.operator))

def exact48598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩]

theorem exact48598RawTermsValid :
    exact48598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48598 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27885⟩⟩) exact48598RawTerms (.finite 8192) 48597 .exactZero (none)

def event48599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event48600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event48601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16023⟩⟩) 0 ⟨15949⟩ 48587

def event48602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16023⟩⟩) 1 ⟨110⟩ 48600

def event48603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16023⟩⟩) (.sum [.predecessor 0 48601 .coefficient, .predecessor 1 48602 .coefficient])

def event48604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16023⟩⟩) (.finite 18)

def event48605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16024⟩⟩) 0 ⟨16023⟩ 48604

def event48606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16024⟩⟩) (.identity (.predecessor 0 48605 .coefficient))

def exact48607RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], []⟩, (1)⟩]

theorem exact48607RawTermsValid :
    exact48607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16024⟩⟩) exact48607RawTerms (.finite 18) 48606 .exactZero (none)

def event48608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact48609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48609RawTermsValid :
    exact48609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48609 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact48609RawTerms .large 48608 .exactZero (none)

def event48610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16025⟩⟩) 0 ⟨6544⟩ 48609

def event48611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16025⟩⟩) 1 ⟨16024⟩ 48607

def event48612 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16025⟩⟩) (.product (.predecessor 0 48610 .coefficient) (.predecessor 1 48611 .coefficient) (⟨false, false, none, none, none⟩))

def event48613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16025⟩⟩, .operator (⟨48609, 0⟩, ⟨48607, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48614RawTermsValid :
    exact48614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16025⟩⟩) exact48614RawTerms .large 48612 .exactZero (none)

def event48615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 48591

def event48616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact48617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact48617RawTermsValid :
    exact48617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact48617RawTerms .large 48616 .exactZero (none)

def event48618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16026⟩⟩) 0 ⟨6697⟩ 48617

def event48619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16026⟩⟩) 1 ⟨16025⟩ 48614

def event48620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16026⟩⟩) (.sum [.predecessor 0 48618 .coefficient, .predecessor 1 48619 .coefficient])

def exact48621RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48621RawTermsValid :
    exact48621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16026⟩⟩) exact48621RawTerms .large 48620 .exactZero (none)

def event48622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27886⟩⟩) 0 ⟨16026⟩ 48621

def event48623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27886⟩⟩) 1 ⟨27885⟩ 48598

def event48624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27886⟩⟩) (.product (.predecessor 0 48622 .coefficient) (.predecessor 1 48623 .coefficient) (⟨false, false, none, none, none⟩))

def event48625 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27886⟩⟩, .operator (⟨48621, 0⟩, ⟨48598, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩)

def event48626 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27886⟩⟩, .operator (⟨48621, 1⟩, ⟨48598, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩)

def event48627 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27886⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27885⟩⟩) ⟨24167⟩ 48595)

def event48628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27886⟩⟩, .relation 48627 0, ⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (-1)⟩)

def exact48629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (-1)⟩]

theorem exact48629RawTermsValid :
    exact48629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27886⟩⟩) exact48629RawTerms .large 48624 .exactZero (none)

def event48630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17173⟩⟩) 0 ⟨15949⟩ 48587

def event48631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17173⟩⟩) (.authority (.programFamilyFact))

def exact48632RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩, (1)⟩]

theorem exact48632RawTermsValid :
    exact48632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48632 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17173⟩⟩) exact48632RawTerms (.finite 18) 48631 .exactZero (none)

def event48633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17175⟩⟩) 0 ⟨6544⟩ 48609

def event48634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17175⟩⟩) 1 ⟨17173⟩ 48632

def event48635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17175⟩⟩) (.product (.predecessor 0 48633 .coefficient) (.predecessor 1 48634 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48636 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17175⟩⟩, .operator (⟨48609, 0⟩, ⟨48632, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48637RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48637RawTermsValid :
    exact48637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48637 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17175⟩⟩) exact48637RawTerms .large 48635 .exactZero (none)

def event48638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6722⟩⟩) 0 ⟨6689⟩ 48591

def event48639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6722⟩⟩) (.authority (.operator))

def eventLeaf3024 : Array AnnotatedEvent := #[
  { event := event48384
    frameStart := 48335 },
  { event := event48385
    frameStart := 48335 },
  { event := event48386
    frameStart := 48335 },
  { event := event48387
    frameStart := 48335 },
  { event := event48388
    frameStart := 48335 },
  { event := event48389
    frameStart := 48335 },
  { event := event48390
    frameStart := 48335 },
  { event := event48391
    frameStart := 48335 },
  { event := event48392
    frameStart := 48335 },
  { event := event48393
    frameStart := 48335 },
  { event := event48394
    frameStart := 48335 },
  { event := event48395
    frameStart := 48335 },
  { event := event48396
    frameStart := 48335 },
  { event := event48397
    frameStart := 48335 },
  { event := event48398
    frameStart := 48335 },
  { event := event48399
    frameStart := 48335 }
]

def eventLeaf3025 : Array AnnotatedEvent := #[
  { event := event48400
    frameStart := 48335 },
  { event := event48401
    frameStart := 48335 },
  { event := event48402
    frameStart := 48335 },
  { event := event48403
    frameStart := 48335 },
  { event := event48404
    frameStart := 48335 },
  { event := event48405
    frameStart := 48335 },
  { event := event48406
    frameStart := 48335 },
  { event := event48407
    frameStart := 48335 },
  { event := event48408
    frameStart := 48335 },
  { event := event48409
    frameStart := 48335 },
  { event := event48410
    frameStart := 48335 },
  { event := event48411
    frameStart := 48335 },
  { event := event48412
    frameStart := 48335 },
  { event := event48413
    frameStart := 48335 },
  { event := event48414
    frameStart := 48335 },
  { event := event48415
    frameStart := 48335 }
]

def eventLeaf3026 : Array AnnotatedEvent := #[
  { event := event48416
    frameStart := 48335 },
  { event := event48417
    frameStart := 48335 },
  { event := event48418
    frameStart := 48335 },
  { event := event48419
    frameStart := 48335 },
  { event := event48420
    frameStart := 48335 },
  { event := event48421
    frameStart := 48335 },
  { event := event48422
    frameStart := 48335 },
  { event := event48423
    frameStart := 48335 },
  { event := event48424
    frameStart := 48335 },
  { event := event48425
    frameStart := 48335 },
  { event := event48426
    frameStart := 48335 },
  { event := event48427
    frameStart := 48335 },
  { event := event48428
    frameStart := 48335 },
  { event := event48429
    frameStart := 48335 },
  { event := event48430
    frameStart := 48335 },
  { event := event48431
    frameStart := 48335 }
]

def eventLeaf3027 : Array AnnotatedEvent := #[
  { event := event48432
    frameStart := 48335 },
  { event := event48433
    frameStart := 48335 },
  { event := event48434
    frameStart := 48335 },
  { event := event48435
    frameStart := 48335 },
  { event := event48436
    frameStart := 48335 },
  { event := event48437
    frameStart := 48335 },
  { event := event48438
    frameStart := 48335 },
  { event := event48439
    frameStart := 0 },
  { event := event48440
    frameStart := 0 },
  { event := event48441
    frameStart := 0 },
  { event := event48442
    frameStart := 0 },
  { event := event48443
    frameStart := 0 },
  { event := event48444
    frameStart := 0 },
  { event := event48445
    frameStart := 0 },
  { event := event48446
    frameStart := 0 },
  { event := event48447
    frameStart := 0 }
]

def eventLeaf3028 : Array AnnotatedEvent := #[
  { event := event48448
    frameStart := 0 },
  { event := event48449
    frameStart := 0 },
  { event := event48450
    frameStart := 0 },
  { event := event48451
    frameStart := 0 },
  { event := event48452
    frameStart := 0 },
  { event := event48453
    frameStart := 0 },
  { event := event48454
    frameStart := 0 },
  { event := event48455
    frameStart := 0 },
  { event := event48456
    frameStart := 0 },
  { event := event48457
    frameStart := 0 },
  { event := event48458
    frameStart := 0 },
  { event := event48459
    frameStart := 0 },
  { event := event48460
    frameStart := 0 },
  { event := event48461
    frameStart := 0 },
  { event := event48462
    frameStart := 0 },
  { event := event48463
    frameStart := 0 }
]

def eventLeaf3029 : Array AnnotatedEvent := #[
  { event := event48464
    frameStart := 0 },
  { event := event48465
    frameStart := 0 },
  { event := event48466
    frameStart := 0 },
  { event := event48467
    frameStart := 0 },
  { event := event48468
    frameStart := 0 },
  { event := event48469
    frameStart := 0 },
  { event := event48470
    frameStart := 0 },
  { event := event48471
    frameStart := 0 },
  { event := event48472
    frameStart := 0 },
  { event := event48473
    frameStart := 0 },
  { event := event48474
    frameStart := 0 },
  { event := event48475
    frameStart := 0 },
  { event := event48476
    frameStart := 0 },
  { event := event48477
    frameStart := 0 },
  { event := event48478
    frameStart := 0 },
  { event := event48479
    frameStart := 0 }
]

def eventLeaf3030 : Array AnnotatedEvent := #[
  { event := event48480
    frameStart := 0 },
  { event := event48481
    frameStart := 0 },
  { event := event48482
    frameStart := 0 },
  { event := event48483
    frameStart := 0 },
  { event := event48484
    frameStart := 0 },
  { event := event48485
    frameStart := 0 },
  { event := event48486
    frameStart := 0 },
  { event := event48487
    frameStart := 0 },
  { event := event48488
    frameStart := 0 },
  { event := event48489
    frameStart := 0 },
  { event := event48490
    frameStart := 0 },
  { event := event48491
    frameStart := 0 },
  { event := event48492
    frameStart := 0 },
  { event := event48493
    frameStart := 48493 },
  { event := event48494
    frameStart := 48493 },
  { event := event48495
    frameStart := 48493 }
]

def eventLeaf3031 : Array AnnotatedEvent := #[
  { event := event48496
    frameStart := 48493 },
  { event := event48497
    frameStart := 48493 },
  { event := event48498
    frameStart := 48493 },
  { event := event48499
    frameStart := 48493 },
  { event := event48500
    frameStart := 48493 },
  { event := event48501
    frameStart := 48493 },
  { event := event48502
    frameStart := 48493 },
  { event := event48503
    frameStart := 48493 },
  { event := event48504
    frameStart := 48493 },
  { event := event48505
    frameStart := 48493 },
  { event := event48506
    frameStart := 48493 },
  { event := event48507
    frameStart := 48493 },
  { event := event48508
    frameStart := 48493 },
  { event := event48509
    frameStart := 48493 },
  { event := event48510
    frameStart := 48493 },
  { event := event48511
    frameStart := 48493 }
]

def eventLeaf3032 : Array AnnotatedEvent := #[
  { event := event48512
    frameStart := 48493 },
  { event := event48513
    frameStart := 48493 },
  { event := event48514
    frameStart := 48493 },
  { event := event48515
    frameStart := 48493 },
  { event := event48516
    frameStart := 48493 },
  { event := event48517
    frameStart := 48493 },
  { event := event48518
    frameStart := 48493 },
  { event := event48519
    frameStart := 48493 },
  { event := event48520
    frameStart := 48493 },
  { event := event48521
    frameStart := 48493 },
  { event := event48522
    frameStart := 48493 },
  { event := event48523
    frameStart := 48493 },
  { event := event48524
    frameStart := 48493 },
  { event := event48525
    frameStart := 48493 },
  { event := event48526
    frameStart := 48493 },
  { event := event48527
    frameStart := 48493 }
]

def eventLeaf3033 : Array AnnotatedEvent := #[
  { event := event48528
    frameStart := 48493 },
  { event := event48529
    frameStart := 48493 },
  { event := event48530
    frameStart := 48493 },
  { event := event48531
    frameStart := 48493 },
  { event := event48532
    frameStart := 48493 },
  { event := event48533
    frameStart := 48493 },
  { event := event48534
    frameStart := 48493 },
  { event := event48535
    frameStart := 48493 },
  { event := event48536
    frameStart := 48493 },
  { event := event48537
    frameStart := 48493 },
  { event := event48538
    frameStart := 48493 },
  { event := event48539
    frameStart := 48493 },
  { event := event48540
    frameStart := 48493 },
  { event := event48541
    frameStart := 48493 },
  { event := event48542
    frameStart := 48493 },
  { event := event48543
    frameStart := 48493 }
]

def eventLeaf3034 : Array AnnotatedEvent := #[
  { event := event48544
    frameStart := 48493 },
  { event := event48545
    frameStart := 48493 },
  { event := event48546
    frameStart := 48493 },
  { event := event48547
    frameStart := 48547 },
  { event := event48548
    frameStart := 48547 },
  { event := event48549
    frameStart := 48547 },
  { event := event48550
    frameStart := 48547 },
  { event := event48551
    frameStart := 48547 },
  { event := event48552
    frameStart := 48547 },
  { event := event48553
    frameStart := 48547 },
  { event := event48554
    frameStart := 48547 },
  { event := event48555
    frameStart := 48547 },
  { event := event48556
    frameStart := 48547 },
  { event := event48557
    frameStart := 48547 },
  { event := event48558
    frameStart := 48547 },
  { event := event48559
    frameStart := 48547 }
]

def eventLeaf3035 : Array AnnotatedEvent := #[
  { event := event48560
    frameStart := 48547 },
  { event := event48561
    frameStart := 48547 },
  { event := event48562
    frameStart := 48547 },
  { event := event48563
    frameStart := 48547 },
  { event := event48564
    frameStart := 48547 },
  { event := event48565
    frameStart := 48547 },
  { event := event48566
    frameStart := 48547 },
  { event := event48567
    frameStart := 48547 },
  { event := event48568
    frameStart := 48547 },
  { event := event48569
    frameStart := 48547 },
  { event := event48570
    frameStart := 48547 },
  { event := event48571
    frameStart := 48547 },
  { event := event48572
    frameStart := 48547 },
  { event := event48573
    frameStart := 48547 },
  { event := event48574
    frameStart := 48547 },
  { event := event48575
    frameStart := 48547 }
]

def eventLeaf3036 : Array AnnotatedEvent := #[
  { event := event48576
    frameStart := 48547 },
  { event := event48577
    frameStart := 48547 },
  { event := event48578
    frameStart := 48547 },
  { event := event48579
    frameStart := 48547 },
  { event := event48580
    frameStart := 48547 },
  { event := event48581
    frameStart := 48547 },
  { event := event48582
    frameStart := 48547 },
  { event := event48583
    frameStart := 48547 },
  { event := event48584
    frameStart := 48547 },
  { event := event48585
    frameStart := 48547 },
  { event := event48586
    frameStart := 48547 },
  { event := event48587
    frameStart := 48547 },
  { event := event48588
    frameStart := 48547 },
  { event := event48589
    frameStart := 48547 },
  { event := event48590
    frameStart := 48547 },
  { event := event48591
    frameStart := 48547 }
]

def eventLeaf3037 : Array AnnotatedEvent := #[
  { event := event48592
    frameStart := 48547 },
  { event := event48593
    frameStart := 48547 },
  { event := event48594
    frameStart := 48547 },
  { event := event48595
    frameStart := 48547 },
  { event := event48596
    frameStart := 48547 },
  { event := event48597
    frameStart := 48547 },
  { event := event48598
    frameStart := 48547 },
  { event := event48599
    frameStart := 48547 },
  { event := event48600
    frameStart := 48547 },
  { event := event48601
    frameStart := 48547 },
  { event := event48602
    frameStart := 48547 },
  { event := event48603
    frameStart := 48547 },
  { event := event48604
    frameStart := 48547 },
  { event := event48605
    frameStart := 48547 },
  { event := event48606
    frameStart := 48547 },
  { event := event48607
    frameStart := 48547 }
]

def eventLeaf3038 : Array AnnotatedEvent := #[
  { event := event48608
    frameStart := 48547 },
  { event := event48609
    frameStart := 48547 },
  { event := event48610
    frameStart := 48547 },
  { event := event48611
    frameStart := 48547 },
  { event := event48612
    frameStart := 48547 },
  { event := event48613
    frameStart := 48547 },
  { event := event48614
    frameStart := 48547 },
  { event := event48615
    frameStart := 48547 },
  { event := event48616
    frameStart := 48547 },
  { event := event48617
    frameStart := 48547 },
  { event := event48618
    frameStart := 48547 },
  { event := event48619
    frameStart := 48547 },
  { event := event48620
    frameStart := 48547 },
  { event := event48621
    frameStart := 48547 },
  { event := event48622
    frameStart := 48547 },
  { event := event48623
    frameStart := 48547 }
]

def eventLeaf3039 : Array AnnotatedEvent := #[
  { event := event48624
    frameStart := 48547 },
  { event := event48625
    frameStart := 48547 },
  { event := event48626
    frameStart := 48547 },
  { event := event48627
    frameStart := 48547 },
  { event := event48628
    frameStart := 48547 },
  { event := event48629
    frameStart := 48547 },
  { event := event48630
    frameStart := 48547 },
  { event := event48631
    frameStart := 48547 },
  { event := event48632
    frameStart := 48547 },
  { event := event48633
    frameStart := 48547 },
  { event := event48634
    frameStart := 48547 },
  { event := event48635
    frameStart := 48547 },
  { event := event48636
    frameStart := 48547 },
  { event := event48637
    frameStart := 48547 },
  { event := event48638
    frameStart := 48547 },
  { event := event48639
    frameStart := 48547 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events189
