import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events271

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event69376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6762⟩⟩) (.identity (.predecessor 0 69375 .coefficient))

def exact69377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact69377RawTermsValid :
    exact69377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6762⟩⟩) exact69377RawTerms .large 69376 .exactZero (none)

def event69378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 0 ⟨6762⟩ 69377

def event69379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 1 ⟨7859⟩ 69374

def event69380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7860⟩⟩) (.product (.predecessor 0 69378 .coefficient) (.predecessor 1 69379 .coefficient) (⟨false, false, none, none, none⟩))

def event69381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7860⟩⟩, .operator (⟨69377, 0⟩, ⟨69374, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact69382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact69382RawTermsValid :
    exact69382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69382 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7860⟩⟩) exact69382RawTerms .large 69380 .exactZero (none)

def event69383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14747⟩⟩) 0 ⟨7860⟩ 69382

def event69384 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14747⟩⟩) 1 ⟨14746⟩ 69359

def event69385 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14747⟩⟩) (.sum [.predecessor 0 69383 .coefficient, .predecessor 1 69384 .coefficient])

def exact69386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69386RawTermsValid :
    exact69386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69386 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14747⟩⟩) exact69386RawTerms .large 69385 .exactZero (none)

def event69387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26218⟩⟩) 0 ⟨14747⟩ 69386

def event69388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26218⟩⟩) 1 ⟨26215⟩ 69343

def event69389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26218⟩⟩) (.product (.predecessor 0 69387 .coefficient) (.predecessor 1 69388 .coefficient) (⟨false, false, none, none, none⟩))

def event69390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26218⟩⟩, .operator (⟨69386, 0⟩, ⟨69343, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩)

def event69391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26218⟩⟩, .operator (⟨69386, 1⟩, ⟨69343, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩)

def event69392 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26218⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26215⟩⟩) ⟨23666⟩ 69340)

def event69393 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26218⟩⟩, .relation 69392 0, ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩)

def exact69394RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩]

theorem exact69394RawTermsValid :
    exact69394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69394 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26218⟩⟩) exact69394RawTerms .large 69389 .exactZero (none)

def event69395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16174⟩⟩) 0 ⟨14634⟩ 69332

def event69396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16174⟩⟩) (.authority (.programFamilyFact))

def exact69397RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact69397RawTermsValid :
    exact69397RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69397 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16174⟩⟩) exact69397RawTerms (.finite 28) 69396 .exactZero (none)

def event69398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16176⟩⟩) 0 ⟨6544⟩ 69354

def event69399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16176⟩⟩) 1 ⟨16174⟩ 69397

def event69400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16176⟩⟩) (.product (.predecessor 0 69398 .coefficient) (.predecessor 1 69399 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69401 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16176⟩⟩, .operator (⟨69354, 0⟩, ⟨69397, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69402RawTermsValid :
    exact69402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16176⟩⟩) exact69402RawTerms .large 69400 .exactZero (none)

def event69403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 69336

def event69404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact69405RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact69405RawTermsValid :
    exact69405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact69405RawTerms .large 69404 .exactZero (none)

def event69406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16177⟩⟩) 0 ⟨6699⟩ 69405

def event69407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16177⟩⟩) 1 ⟨16176⟩ 69402

def event69408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16177⟩⟩) (.sum [.predecessor 0 69406 .coefficient, .predecessor 1 69407 .coefficient])

def exact69409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69409RawTermsValid :
    exact69409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16177⟩⟩) exact69409RawTerms .large 69408 .exactZero (none)

def event69410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26219⟩⟩) 0 ⟨16177⟩ 69409

def event69411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26219⟩⟩) 1 ⟨26218⟩ 69394

def event69412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26219⟩⟩) (.sum [.predecessor 0 69410 .coefficient, .predecessor 1 69411 .coefficient])

def exact69413RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69413RawTermsValid :
    exact69413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26219⟩⟩) exact69413RawTerms .large 69412 .exactZero (none)

def event69414 : Event := .preFoldPolynomial 69413 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact69415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event69415 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26219⟩⟩) 69414 exact69415RawTerms .large 69412 .exactZero (none)

def event69416 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14634⟩⟩) ⟨⟨112⟩, ⟨17⟩, ⟨109⟩⟩ ⟨69250, 69416⟩

def event69417 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19671⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (1) 0 2 (.universal 69416 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (none) 69415)

def event69418 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19671⟩⟩, .relation 69417 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩)

def event69419 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19671⟩⟩, .relation 69417 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩)

def event69420 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19671⟩⟩, .relation 69417 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩)

def event69421 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19671⟩⟩, .relation 69417 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact69422RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69422RawTermsValid :
    exact69422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19671⟩⟩) exact69422RawTerms .large 69246 (.finite 1811303510016) (some (69248))

def event69423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26217⟩⟩) 0 ⟨19671⟩ 69422

def event69424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26217⟩⟩) 1 ⟨26216⟩ 69236

def event69425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26217⟩⟩) (.sum [.predecessor 0 69423 .coefficient, .predecessor 1 69424 .coefficient])

def event69426 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26217⟩⟩, .operator (⟨69422, 2⟩, ⟨69236, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩)

def event69427 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26217⟩⟩, .operator (⟨69422, 1⟩, ⟨69236, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩)

def event69428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26217⟩⟩) (.sum [.result 69422 .summary, .result 69236 .summary])

def exact69429RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69429RawTermsValid :
    exact69429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26217⟩⟩) exact69429RawTerms .large 69425 (.finite 352091253649408) (some (69428))

def event69430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28289⟩⟩) 0 ⟨26217⟩ 69429

def event69431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28289⟩⟩) 1 ⟨28287⟩ 69152

def event69432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28289⟩⟩) (.product (.predecessor 0 69430 .coefficient) (.predecessor 1 69431 .coefficient) (⟨false, false, none, none, none⟩))

def event69433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28289⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) [⟨.result 69152 .coefficient, false, none⟩])

def event69434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28289⟩⟩) (.product (.result 69429 .summary) (.transfer 69433) (⟨false, false, none, none, none⟩))

def event69435 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28289⟩⟩, .operator (⟨69429, 0⟩, ⟨69152, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩)

def event69436 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28289⟩⟩, .operator (⟨69429, 1⟩, ⟨69152, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩)

def event69437 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28289⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28287⟩⟩) ⟨24285⟩ 69149)

def event69438 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28289⟩⟩, .relation 69437 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (-1)⟩)

def exact69439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (-1)⟩]

theorem exact69439RawTermsValid :
    exact69439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28289⟩⟩) exact69439RawTerms .large 69432 (.finite 1292180534353385750528) (some (69434))

def event69440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21684⟩⟩) 0 ⟨16175⟩ 3287

def event69441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21684⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact69442RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩]

theorem exact69442RawTermsValid :
    exact69442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69442 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21684⟩⟩) exact69442RawTerms (.finite 136065468) 69441 .exactZero (none)

def event69443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21686⟩⟩) 0 ⟨21684⟩ 69442

def event69444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21686⟩⟩) 1 ⟨2348⟩ 4

def event69445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21686⟩⟩) (.scale (.predecessor 0 69443 .coefficient) (.value (.predecessor 1 69444 .coefficient)))

def exact69446RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩]

theorem exact69446RawTermsValid :
    exact69446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21686⟩⟩) exact69446RawTerms (.finite 136065468) 69445 .exactZero (none)

def event69447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21687⟩⟩) 0 ⟨5535⟩ 65387

def event69448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21687⟩⟩) 1 ⟨21686⟩ 69446

def event69449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21687⟩⟩) (.product (.predecessor 0 69447 .coefficient) (.predecessor 1 69448 .coefficient) (⟨false, false, none, none, none⟩))

def event69450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21687⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩) [⟨.result 69442 .coefficient, false, none⟩])

def event69451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21687⟩⟩) (.product (.result 65387 .summary) (.transfer 69450) (⟨false, false, none, none, none⟩))

def event69452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21687⟩⟩, .operator (⟨65387, 0⟩, ⟨69446, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩)

def event69453 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21685⟩⟩)

def event69454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69455 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69457 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69459 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69461 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69461

def event69463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69459

def event69464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69462 .coefficient) (.value (.predecessor 1 69463 .coefficient)))

def event69465 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69465

def event69467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69457

def event69468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69466 .coefficient, .predecessor 1 69467 .coefficient])

def event69469 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69469

def event69471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69455

def event69472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69471 .coefficient))

def event69473 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 69473

def event69475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact69476RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact69476RawTermsValid :
    exact69476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact69476RawTerms (.finite 28) 69475 .exactZero (none)

def event69477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 69473

def event69478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact69479RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69479RawTermsValid :
    exact69479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact69479RawTerms (.finite 28) 69478 .exactZero (none)

def event69480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 69479

def event69481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 69476

def event69482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 69480 .coefficient) (.predecessor 1 69481 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩) [⟨.result 69479 .coefficient, true, some 1⟩, ⟨.result 69476 .coefficient, true, some 1⟩])

def event69484 : Event := .survivorFold (1) 69483

def exact69485RawTerms : List Term := []

theorem exact69485RawTermsValid :
    exact69485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact69485RawTerms (.finite 784) 69482 (.finite 784) (some (69483))

def event69486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 69485

def event69487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 69486 .coefficient))

def event69488 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event69489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16174⟩⟩) 0 ⟨14634⟩ 69488

def event69490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16174⟩⟩) (.authority (.programFamilyFact))

def exact69491RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact69491RawTermsValid :
    exact69491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16174⟩⟩) exact69491RawTerms (.finite 28) 69490 .exactZero (none)

def event69492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16175⟩⟩) 0 ⟨16174⟩ 69491

def event69493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.identity (.predecessor 0 69492 .coefficient))

def event69494 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.finite 28)

def event69495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21684⟩⟩) 0 ⟨16175⟩ 69494

def event69496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21684⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact69497RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩]

theorem exact69497RawTermsValid :
    exact69497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21684⟩⟩) exact69497RawTerms (.finite 136065468) 69496 .exactZero (none)

def event69498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact69499RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact69499RawTermsValid :
    exact69499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69499 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact69499RawTerms .large 69498 .exactZero (none)

def event69500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21685⟩⟩) 0 ⟨6⟩ 69499

def event69501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21685⟩⟩) 1 ⟨21684⟩ 69497

def event69502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21685⟩⟩) (.product (.predecessor 0 69500 .coefficient) (.predecessor 1 69501 .coefficient) (⟨false, false, none, none, none⟩))

def event69503 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21685⟩⟩, .operator (⟨69499, 0⟩, ⟨69497, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩)

def exact69504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩]

theorem exact69504RawTermsValid :
    exact69504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21685⟩⟩) exact69504RawTerms .large 69502 .exactZero (none)

def event69505 : Event := .preFoldPolynomial 69504 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩] .exactZero none

def exact69506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩, (1)⟩]

def event69506 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21685⟩⟩) 69505 exact69506RawTerms .large 69502 .exactZero (none)

def event69507 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28292⟩⟩)

def event69508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69509 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69513 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69515 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69515

def event69517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69513

def event69518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69516 .coefficient) (.value (.predecessor 1 69517 .coefficient)))

def event69519 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69519

def event69521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69511

def event69522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69520 .coefficient, .predecessor 1 69521 .coefficient])

def event69523 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69524 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69523

def event69525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69509

def event69526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69525 .coefficient))

def event69527 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 69527

def event69529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact69530RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact69530RawTermsValid :
    exact69530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact69530RawTerms (.finite 28) 69529 .exactZero (none)

def event69531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 69527

def event69532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact69533RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69533RawTermsValid :
    exact69533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact69533RawTerms (.finite 28) 69532 .exactZero (none)

def event69534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 69533

def event69535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 69530

def event69536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 69534 .coefficient) (.predecessor 1 69535 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69537 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14633⟩⟩, .operator (⟨69533, 0⟩, ⟨69530, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩)

def exact69538RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69538RawTermsValid :
    exact69538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact69538RawTerms (.finite 784) 69536 .exactZero (none)

def event69539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 69538

def event69540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 69539 .coefficient))

def event69541 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event69542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16174⟩⟩) 0 ⟨14634⟩ 69541

def event69543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16174⟩⟩) (.authority (.programFamilyFact))

def exact69544RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact69544RawTermsValid :
    exact69544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16174⟩⟩) exact69544RawTerms (.finite 28) 69543 .exactZero (none)

def event69545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16175⟩⟩) 0 ⟨16174⟩ 69544

def event69546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.identity (.predecessor 0 69545 .coefficient))

def event69547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16175⟩⟩) (.finite 28)

def event69548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24283⟩⟩) 0 ⟨16175⟩ 69547

def event69549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.authority (.programFamilyFact))

def event69550 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.finite 3720)

def event69551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event69552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24285⟩⟩) 0 ⟨6689⟩ 69551

def event69553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24285⟩⟩) 1 ⟨24283⟩ 69550

def event69554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24285⟩⟩) (.authority (.operator))

def exact69555RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩]

theorem exact69555RawTermsValid :
    exact69555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69555 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24285⟩⟩) exact69555RawTerms .large 69554 .exactZero (none)

def event69556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28287⟩⟩) 0 ⟨24285⟩ 69555

def event69557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28287⟩⟩) (.authority (.operator))

def exact69558RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩]

theorem exact69558RawTermsValid :
    exact69558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69558 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28287⟩⟩) exact69558RawTerms (.finite 8192) 69557 .exactZero (none)

def event69559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event69560 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event69561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16214⟩⟩) 0 ⟨16175⟩ 69547

def event69562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16214⟩⟩) 1 ⟨110⟩ 69560

def event69563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16214⟩⟩) (.sum [.predecessor 0 69561 .coefficient, .predecessor 1 69562 .coefficient])

def event69564 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16214⟩⟩) (.finite 28)

def event69565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16215⟩⟩) 0 ⟨16214⟩ 69564

def event69566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16215⟩⟩) (.identity (.predecessor 0 69565 .coefficient))

def exact69567RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], []⟩, (1)⟩]

theorem exact69567RawTermsValid :
    exact69567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69567 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16215⟩⟩) exact69567RawTerms (.finite 28) 69566 .exactZero (none)

def event69568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact69569RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69569RawTermsValid :
    exact69569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact69569RawTerms .large 69568 .exactZero (none)

def event69570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16216⟩⟩) 0 ⟨6544⟩ 69569

def event69571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16216⟩⟩) 1 ⟨16215⟩ 69567

def event69572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16216⟩⟩) (.product (.predecessor 0 69570 .coefficient) (.predecessor 1 69571 .coefficient) (⟨false, false, none, none, none⟩))

def event69573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16216⟩⟩, .operator (⟨69569, 0⟩, ⟨69567, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69574RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69574RawTermsValid :
    exact69574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16216⟩⟩) exact69574RawTerms .large 69572 .exactZero (none)

def event69575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 69551

def event69576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact69577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact69577RawTermsValid :
    exact69577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact69577RawTerms .large 69576 .exactZero (none)

def event69578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16217⟩⟩) 0 ⟨6699⟩ 69577

def event69579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16217⟩⟩) 1 ⟨16216⟩ 69574

def event69580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16217⟩⟩) (.sum [.predecessor 0 69578 .coefficient, .predecessor 1 69579 .coefficient])

def exact69581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69581RawTermsValid :
    exact69581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16217⟩⟩) exact69581RawTerms .large 69580 .exactZero (none)

def event69582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28288⟩⟩) 0 ⟨16217⟩ 69581

def event69583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28288⟩⟩) 1 ⟨28287⟩ 69558

def event69584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28288⟩⟩) (.product (.predecessor 0 69582 .coefficient) (.predecessor 1 69583 .coefficient) (⟨false, false, none, none, none⟩))

def event69585 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28288⟩⟩, .operator (⟨69581, 0⟩, ⟨69558, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩)

def event69586 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28288⟩⟩, .operator (⟨69581, 1⟩, ⟨69558, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩)

def event69587 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28288⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28287⟩⟩) ⟨24285⟩ 69555)

def event69588 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28288⟩⟩, .relation 69587 0, ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (-1)⟩)

def exact69589RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (-1)⟩]

theorem exact69589RawTermsValid :
    exact69589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69589 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28288⟩⟩) exact69589RawTerms .large 69584 .exactZero (none)

def event69590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18327⟩⟩) 0 ⟨16175⟩ 69547

def event69591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18327⟩⟩) (.authority (.programFamilyFact))

def exact69592RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], []⟩, (1)⟩]

theorem exact69592RawTermsValid :
    exact69592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18327⟩⟩) exact69592RawTerms (.finite 62) 69591 .exactZero (none)

def event69593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18338⟩⟩) 0 ⟨6544⟩ 69569

def event69594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18338⟩⟩) 1 ⟨18327⟩ 69592

def event69595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18338⟩⟩) (.product (.predecessor 0 69593 .coefficient) (.predecessor 1 69594 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18338⟩⟩, .operator (⟨69569, 0⟩, ⟨69592, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69597RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69597RawTermsValid :
    exact69597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18338⟩⟩) exact69597RawTerms .large 69595 .exactZero (none)

def event69598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 69551

def event69599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact69600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact69600RawTermsValid :
    exact69600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69600 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact69600RawTerms .large 69599 .exactZero (none)

def event69601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18339⟩⟩) 0 ⟨6727⟩ 69600

def event69602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18339⟩⟩) 1 ⟨18338⟩ 69597

def event69603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18339⟩⟩) (.sum [.predecessor 0 69601 .coefficient, .predecessor 1 69602 .coefficient])

def exact69604RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69604RawTermsValid :
    exact69604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69604 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18339⟩⟩) exact69604RawTerms .large 69603 .exactZero (none)

def event69605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28292⟩⟩) 0 ⟨18339⟩ 69604

def event69606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28292⟩⟩) 1 ⟨28288⟩ 69589

def event69607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28292⟩⟩) (.sum [.predecessor 0 69605 .coefficient, .predecessor 1 69606 .coefficient])

def exact69608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69608RawTermsValid :
    exact69608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69608 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28292⟩⟩) exact69608RawTerms .large 69607 .exactZero (none)

def event69609 : Event := .preFoldPolynomial 69608 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact69610RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event69610 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28292⟩⟩) 69609 exact69610RawTerms .large 69607 .exactZero (none)

def event69611 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16175⟩⟩) ⟨⟨140⟩, ⟨48⟩, ⟨109⟩⟩ ⟨69453, 69611⟩

def event69612 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21687⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩) (1) 0 2 (.universal 69611 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩) (none) 69610)

def event69613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21687⟩⟩, .relation 69612 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩)

def event69614 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21687⟩⟩, .relation 69612 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩)

def event69615 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21687⟩⟩, .relation 69612 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩)

def event69616 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21687⟩⟩, .relation 69612 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact69617RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69617RawTermsValid :
    exact69617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21687⟩⟩) exact69617RawTerms .large 69449 (.finite 1811303510016) (some (69451))

def event69618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28290⟩⟩) 0 ⟨21687⟩ 69617

def event69619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28290⟩⟩) 1 ⟨28289⟩ 69439

def event69620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28290⟩⟩) (.sum [.predecessor 0 69618 .coefficient, .predecessor 1 69619 .coefficient])

def event69621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28290⟩⟩, .operator (⟨69617, 0⟩, ⟨69439, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩)

def event69622 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28290⟩⟩, .operator (⟨69617, 2⟩, ⟨69439, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (-1)⟩)

def event69623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28290⟩⟩) (.sum [.result 69617 .summary, .result 69439 .summary])

def exact69624RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69624RawTermsValid :
    exact69624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28290⟩⟩) exact69624RawTerms .large 69620 (.finite 1292180536164689260544) (some (69623))

def event69625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24220⟩⟩) 0 ⟨16056⟩ 3310

def event69626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.authority (.programFamilyFact))

def event69627 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.finite 3720)

def event69628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24222⟩⟩) 0 ⟨6689⟩ 5477

def event69629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24222⟩⟩) 1 ⟨24220⟩ 69627

def event69630 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24222⟩⟩) (.authority (.operator))

def exact69631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩]

theorem exact69631RawTermsValid :
    exact69631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69631 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24222⟩⟩) exact69631RawTerms .large 69630 .exactZero (none)

def eventLeaf4336 : Array AnnotatedEvent := #[
  { event := event69376
    frameStart := 69298 },
  { event := event69377
    frameStart := 69298 },
  { event := event69378
    frameStart := 69298 },
  { event := event69379
    frameStart := 69298 },
  { event := event69380
    frameStart := 69298 },
  { event := event69381
    frameStart := 69298 },
  { event := event69382
    frameStart := 69298 },
  { event := event69383
    frameStart := 69298 },
  { event := event69384
    frameStart := 69298 },
  { event := event69385
    frameStart := 69298 },
  { event := event69386
    frameStart := 69298 },
  { event := event69387
    frameStart := 69298 },
  { event := event69388
    frameStart := 69298 },
  { event := event69389
    frameStart := 69298 },
  { event := event69390
    frameStart := 69298 },
  { event := event69391
    frameStart := 69298 }
]

def eventLeaf4337 : Array AnnotatedEvent := #[
  { event := event69392
    frameStart := 69298 },
  { event := event69393
    frameStart := 69298 },
  { event := event69394
    frameStart := 69298 },
  { event := event69395
    frameStart := 69298 },
  { event := event69396
    frameStart := 69298 },
  { event := event69397
    frameStart := 69298 },
  { event := event69398
    frameStart := 69298 },
  { event := event69399
    frameStart := 69298 },
  { event := event69400
    frameStart := 69298 },
  { event := event69401
    frameStart := 69298 },
  { event := event69402
    frameStart := 69298 },
  { event := event69403
    frameStart := 69298 },
  { event := event69404
    frameStart := 69298 },
  { event := event69405
    frameStart := 69298 },
  { event := event69406
    frameStart := 69298 },
  { event := event69407
    frameStart := 69298 }
]

def eventLeaf4338 : Array AnnotatedEvent := #[
  { event := event69408
    frameStart := 69298 },
  { event := event69409
    frameStart := 69298 },
  { event := event69410
    frameStart := 69298 },
  { event := event69411
    frameStart := 69298 },
  { event := event69412
    frameStart := 69298 },
  { event := event69413
    frameStart := 69298 },
  { event := event69414
    frameStart := 69298 },
  { event := event69415
    frameStart := 69298 },
  { event := event69416
    frameStart := 0 },
  { event := event69417
    frameStart := 0 },
  { event := event69418
    frameStart := 0 },
  { event := event69419
    frameStart := 0 },
  { event := event69420
    frameStart := 0 },
  { event := event69421
    frameStart := 0 },
  { event := event69422
    frameStart := 0 },
  { event := event69423
    frameStart := 0 }
]

def eventLeaf4339 : Array AnnotatedEvent := #[
  { event := event69424
    frameStart := 0 },
  { event := event69425
    frameStart := 0 },
  { event := event69426
    frameStart := 0 },
  { event := event69427
    frameStart := 0 },
  { event := event69428
    frameStart := 0 },
  { event := event69429
    frameStart := 0 },
  { event := event69430
    frameStart := 0 },
  { event := event69431
    frameStart := 0 },
  { event := event69432
    frameStart := 0 },
  { event := event69433
    frameStart := 0 },
  { event := event69434
    frameStart := 0 },
  { event := event69435
    frameStart := 0 },
  { event := event69436
    frameStart := 0 },
  { event := event69437
    frameStart := 0 },
  { event := event69438
    frameStart := 0 },
  { event := event69439
    frameStart := 0 }
]

def eventLeaf4340 : Array AnnotatedEvent := #[
  { event := event69440
    frameStart := 0 },
  { event := event69441
    frameStart := 0 },
  { event := event69442
    frameStart := 0 },
  { event := event69443
    frameStart := 0 },
  { event := event69444
    frameStart := 0 },
  { event := event69445
    frameStart := 0 },
  { event := event69446
    frameStart := 0 },
  { event := event69447
    frameStart := 0 },
  { event := event69448
    frameStart := 0 },
  { event := event69449
    frameStart := 0 },
  { event := event69450
    frameStart := 0 },
  { event := event69451
    frameStart := 0 },
  { event := event69452
    frameStart := 0 },
  { event := event69453
    frameStart := 69453 },
  { event := event69454
    frameStart := 69453 },
  { event := event69455
    frameStart := 69453 }
]

def eventLeaf4341 : Array AnnotatedEvent := #[
  { event := event69456
    frameStart := 69453 },
  { event := event69457
    frameStart := 69453 },
  { event := event69458
    frameStart := 69453 },
  { event := event69459
    frameStart := 69453 },
  { event := event69460
    frameStart := 69453 },
  { event := event69461
    frameStart := 69453 },
  { event := event69462
    frameStart := 69453 },
  { event := event69463
    frameStart := 69453 },
  { event := event69464
    frameStart := 69453 },
  { event := event69465
    frameStart := 69453 },
  { event := event69466
    frameStart := 69453 },
  { event := event69467
    frameStart := 69453 },
  { event := event69468
    frameStart := 69453 },
  { event := event69469
    frameStart := 69453 },
  { event := event69470
    frameStart := 69453 },
  { event := event69471
    frameStart := 69453 }
]

def eventLeaf4342 : Array AnnotatedEvent := #[
  { event := event69472
    frameStart := 69453 },
  { event := event69473
    frameStart := 69453 },
  { event := event69474
    frameStart := 69453 },
  { event := event69475
    frameStart := 69453 },
  { event := event69476
    frameStart := 69453 },
  { event := event69477
    frameStart := 69453 },
  { event := event69478
    frameStart := 69453 },
  { event := event69479
    frameStart := 69453 },
  { event := event69480
    frameStart := 69453 },
  { event := event69481
    frameStart := 69453 },
  { event := event69482
    frameStart := 69453 },
  { event := event69483
    frameStart := 69453 },
  { event := event69484
    frameStart := 69453 },
  { event := event69485
    frameStart := 69453 },
  { event := event69486
    frameStart := 69453 },
  { event := event69487
    frameStart := 69453 }
]

def eventLeaf4343 : Array AnnotatedEvent := #[
  { event := event69488
    frameStart := 69453 },
  { event := event69489
    frameStart := 69453 },
  { event := event69490
    frameStart := 69453 },
  { event := event69491
    frameStart := 69453 },
  { event := event69492
    frameStart := 69453 },
  { event := event69493
    frameStart := 69453 },
  { event := event69494
    frameStart := 69453 },
  { event := event69495
    frameStart := 69453 },
  { event := event69496
    frameStart := 69453 },
  { event := event69497
    frameStart := 69453 },
  { event := event69498
    frameStart := 69453 },
  { event := event69499
    frameStart := 69453 },
  { event := event69500
    frameStart := 69453 },
  { event := event69501
    frameStart := 69453 },
  { event := event69502
    frameStart := 69453 },
  { event := event69503
    frameStart := 69453 }
]

def eventLeaf4344 : Array AnnotatedEvent := #[
  { event := event69504
    frameStart := 69453 },
  { event := event69505
    frameStart := 69453 },
  { event := event69506
    frameStart := 69453 },
  { event := event69507
    frameStart := 69507 },
  { event := event69508
    frameStart := 69507 },
  { event := event69509
    frameStart := 69507 },
  { event := event69510
    frameStart := 69507 },
  { event := event69511
    frameStart := 69507 },
  { event := event69512
    frameStart := 69507 },
  { event := event69513
    frameStart := 69507 },
  { event := event69514
    frameStart := 69507 },
  { event := event69515
    frameStart := 69507 },
  { event := event69516
    frameStart := 69507 },
  { event := event69517
    frameStart := 69507 },
  { event := event69518
    frameStart := 69507 },
  { event := event69519
    frameStart := 69507 }
]

def eventLeaf4345 : Array AnnotatedEvent := #[
  { event := event69520
    frameStart := 69507 },
  { event := event69521
    frameStart := 69507 },
  { event := event69522
    frameStart := 69507 },
  { event := event69523
    frameStart := 69507 },
  { event := event69524
    frameStart := 69507 },
  { event := event69525
    frameStart := 69507 },
  { event := event69526
    frameStart := 69507 },
  { event := event69527
    frameStart := 69507 },
  { event := event69528
    frameStart := 69507 },
  { event := event69529
    frameStart := 69507 },
  { event := event69530
    frameStart := 69507 },
  { event := event69531
    frameStart := 69507 },
  { event := event69532
    frameStart := 69507 },
  { event := event69533
    frameStart := 69507 },
  { event := event69534
    frameStart := 69507 },
  { event := event69535
    frameStart := 69507 }
]

def eventLeaf4346 : Array AnnotatedEvent := #[
  { event := event69536
    frameStart := 69507 },
  { event := event69537
    frameStart := 69507 },
  { event := event69538
    frameStart := 69507 },
  { event := event69539
    frameStart := 69507 },
  { event := event69540
    frameStart := 69507 },
  { event := event69541
    frameStart := 69507 },
  { event := event69542
    frameStart := 69507 },
  { event := event69543
    frameStart := 69507 },
  { event := event69544
    frameStart := 69507 },
  { event := event69545
    frameStart := 69507 },
  { event := event69546
    frameStart := 69507 },
  { event := event69547
    frameStart := 69507 },
  { event := event69548
    frameStart := 69507 },
  { event := event69549
    frameStart := 69507 },
  { event := event69550
    frameStart := 69507 },
  { event := event69551
    frameStart := 69507 }
]

def eventLeaf4347 : Array AnnotatedEvent := #[
  { event := event69552
    frameStart := 69507 },
  { event := event69553
    frameStart := 69507 },
  { event := event69554
    frameStart := 69507 },
  { event := event69555
    frameStart := 69507 },
  { event := event69556
    frameStart := 69507 },
  { event := event69557
    frameStart := 69507 },
  { event := event69558
    frameStart := 69507 },
  { event := event69559
    frameStart := 69507 },
  { event := event69560
    frameStart := 69507 },
  { event := event69561
    frameStart := 69507 },
  { event := event69562
    frameStart := 69507 },
  { event := event69563
    frameStart := 69507 },
  { event := event69564
    frameStart := 69507 },
  { event := event69565
    frameStart := 69507 },
  { event := event69566
    frameStart := 69507 },
  { event := event69567
    frameStart := 69507 }
]

def eventLeaf4348 : Array AnnotatedEvent := #[
  { event := event69568
    frameStart := 69507 },
  { event := event69569
    frameStart := 69507 },
  { event := event69570
    frameStart := 69507 },
  { event := event69571
    frameStart := 69507 },
  { event := event69572
    frameStart := 69507 },
  { event := event69573
    frameStart := 69507 },
  { event := event69574
    frameStart := 69507 },
  { event := event69575
    frameStart := 69507 },
  { event := event69576
    frameStart := 69507 },
  { event := event69577
    frameStart := 69507 },
  { event := event69578
    frameStart := 69507 },
  { event := event69579
    frameStart := 69507 },
  { event := event69580
    frameStart := 69507 },
  { event := event69581
    frameStart := 69507 },
  { event := event69582
    frameStart := 69507 },
  { event := event69583
    frameStart := 69507 }
]

def eventLeaf4349 : Array AnnotatedEvent := #[
  { event := event69584
    frameStart := 69507 },
  { event := event69585
    frameStart := 69507 },
  { event := event69586
    frameStart := 69507 },
  { event := event69587
    frameStart := 69507 },
  { event := event69588
    frameStart := 69507 },
  { event := event69589
    frameStart := 69507 },
  { event := event69590
    frameStart := 69507 },
  { event := event69591
    frameStart := 69507 },
  { event := event69592
    frameStart := 69507 },
  { event := event69593
    frameStart := 69507 },
  { event := event69594
    frameStart := 69507 },
  { event := event69595
    frameStart := 69507 },
  { event := event69596
    frameStart := 69507 },
  { event := event69597
    frameStart := 69507 },
  { event := event69598
    frameStart := 69507 },
  { event := event69599
    frameStart := 69507 }
]

def eventLeaf4350 : Array AnnotatedEvent := #[
  { event := event69600
    frameStart := 69507 },
  { event := event69601
    frameStart := 69507 },
  { event := event69602
    frameStart := 69507 },
  { event := event69603
    frameStart := 69507 },
  { event := event69604
    frameStart := 69507 },
  { event := event69605
    frameStart := 69507 },
  { event := event69606
    frameStart := 69507 },
  { event := event69607
    frameStart := 69507 },
  { event := event69608
    frameStart := 69507 },
  { event := event69609
    frameStart := 69507 },
  { event := event69610
    frameStart := 69507 },
  { event := event69611
    frameStart := 0 },
  { event := event69612
    frameStart := 0 },
  { event := event69613
    frameStart := 0 },
  { event := event69614
    frameStart := 0 },
  { event := event69615
    frameStart := 0 }
]

def eventLeaf4351 : Array AnnotatedEvent := #[
  { event := event69616
    frameStart := 0 },
  { event := event69617
    frameStart := 0 },
  { event := event69618
    frameStart := 0 },
  { event := event69619
    frameStart := 0 },
  { event := event69620
    frameStart := 0 },
  { event := event69621
    frameStart := 0 },
  { event := event69622
    frameStart := 0 },
  { event := event69623
    frameStart := 0 },
  { event := event69624
    frameStart := 0 },
  { event := event69625
    frameStart := 0 },
  { event := event69626
    frameStart := 0 },
  { event := event69627
    frameStart := 0 },
  { event := event69628
    frameStart := 0 },
  { event := event69629
    frameStart := 0 },
  { event := event69630
    frameStart := 0 },
  { event := event69631
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events271
