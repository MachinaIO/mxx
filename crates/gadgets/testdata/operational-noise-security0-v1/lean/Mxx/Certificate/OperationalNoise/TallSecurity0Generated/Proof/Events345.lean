import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events345

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event88320 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24914⟩⟩, .relation 88319 0, ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (-1)⟩)

def exact88321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (-1)⟩]

theorem exact88321RawTermsValid :
    exact88321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24914⟩⟩) exact88321RawTerms .large 88316 .exactZero (none)

def event88322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14792⟩⟩) 0 ⟨10482⟩ 88261

def event88323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14792⟩⟩) (.authority (.programFamilyFact))

def exact88324RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact88324RawTermsValid :
    exact88324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14792⟩⟩) exact88324RawTerms (.finite 2) 88323 .exactZero (none)

def event88325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14794⟩⟩) 0 ⟨6544⟩ 88283

def event88326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14794⟩⟩) 1 ⟨14792⟩ 88324

def event88327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14794⟩⟩) (.product (.predecessor 0 88325 .coefficient) (.predecessor 1 88326 .coefficient) (⟨false, true, none, none, some 1⟩))

def event88328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14794⟩⟩, .operator (⟨88283, 0⟩, ⟨88324, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88329RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88329RawTermsValid :
    exact88329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14794⟩⟩) exact88329RawTerms .large 88327 .exactZero (none)

def event88330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 88265

def event88331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact88332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact88332RawTermsValid :
    exact88332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact88332RawTerms .large 88331 .exactZero (none)

def event88333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14795⟩⟩) 0 ⟨6690⟩ 88332

def event88334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14795⟩⟩) 1 ⟨14794⟩ 88329

def event88335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14795⟩⟩) (.sum [.predecessor 0 88333 .coefficient, .predecessor 1 88334 .coefficient])

def exact88336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88336RawTermsValid :
    exact88336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14795⟩⟩) exact88336RawTerms .large 88335 .exactZero (none)

def event88337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24915⟩⟩) 0 ⟨14795⟩ 88336

def event88338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24915⟩⟩) 1 ⟨24914⟩ 88321

def event88339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24915⟩⟩) (.sum [.predecessor 0 88337 .coefficient, .predecessor 1 88338 .coefficient])

def exact88340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88340RawTermsValid :
    exact88340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88340 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24915⟩⟩) exact88340RawTerms .large 88339 .exactZero (none)

def event88341 : Event := .preFoldPolynomial 88340 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact88342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event88342 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨24915⟩⟩) 88341 exact88342RawTerms .large 88339 .exactZero (none)

def event88343 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10482⟩⟩) ⟨⟨103⟩, ⟨7⟩, ⟨109⟩⟩ ⟨88179, 88343⟩

def event88344 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19027⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩) (1) 0 2 (.universal 88343 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩) (none) 88342)

def event88345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19027⟩⟩, .relation 88344 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩)

def event88346 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19027⟩⟩, .relation 88344 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩)

def event88347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19027⟩⟩, .relation 88344 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩)

def event88348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19027⟩⟩, .relation 88344 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact88349RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88349RawTermsValid :
    exact88349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88349 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19027⟩⟩) exact88349RawTerms .large 88175 (.finite 1811303510016) (some (88177))

def event88350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24913⟩⟩) 0 ⟨19027⟩ 88349

def event88351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24913⟩⟩) 1 ⟨24912⟩ 88165

def event88352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24913⟩⟩) (.sum [.predecessor 0 88350 .coefficient, .predecessor 1 88351 .coefficient])

def event88353 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24913⟩⟩, .operator (⟨88349, 2⟩, ⟨88165, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (-1)⟩)

def event88354 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24913⟩⟩, .operator (⟨88349, 1⟩, ⟨88165, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩)

def event88355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24913⟩⟩) (.sum [.result 88349 .summary, .result 88165 .summary])

def exact88356RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88356RawTermsValid :
    exact88356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88356 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24913⟩⟩) exact88356RawTerms .large 88352 (.finite 352011863863296) (some (88355))

def event88357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26360⟩⟩) 0 ⟨24913⟩ 88356

def event88358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26360⟩⟩) 1 ⟨26358⟩ 88081

def event88359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26360⟩⟩) (.product (.predecessor 0 88357 .coefficient) (.predecessor 1 88358 .coefficient) (⟨false, false, none, none, none⟩))

def event88360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26360⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩) [⟨.result 88081 .coefficient, false, none⟩])

def event88361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26360⟩⟩) (.product (.result 88356 .summary) (.transfer 88360) (⟨false, false, none, none, none⟩))

def event88362 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26360⟩⟩, .operator (⟨88356, 0⟩, ⟨88081, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩)

def event88363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26360⟩⟩, .operator (⟨88356, 1⟩, ⟨88081, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩)

def event88364 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26360⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26358⟩⟩) ⟨23721⟩ 88078)

def event88365 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26360⟩⟩, .relation 88364 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (-1)⟩)

def exact88366RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (-1)⟩]

theorem exact88366RawTermsValid :
    exact88366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88366 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26360⟩⟩) exact88366RawTerms .large 88359 (.finite 1291889172568118132736) (some (88361))

def event88367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20392⟩⟩) 0 ⟨14793⟩ 4236

def event88368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20392⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact88369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩]

theorem exact88369RawTermsValid :
    exact88369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20392⟩⟩) exact88369RawTerms (.finite 136065468) 88368 .exactZero (none)

def event88370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20394⟩⟩) 0 ⟨20392⟩ 88369

def event88371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20394⟩⟩) 1 ⟨2348⟩ 4

def event88372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20394⟩⟩) (.scale (.predecessor 0 88370 .coefficient) (.value (.predecessor 1 88371 .coefficient)))

def exact88373RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩]

theorem exact88373RawTermsValid :
    exact88373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20394⟩⟩) exact88373RawTerms (.finite 136065468) 88372 .exactZero (none)

def event88374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20395⟩⟩) 0 ⟨5541⟩ 80012

def event88375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20395⟩⟩) 1 ⟨20394⟩ 88373

def event88376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20395⟩⟩) (.product (.predecessor 0 88374 .coefficient) (.predecessor 1 88375 .coefficient) (⟨false, false, none, none, none⟩))

def event88377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20395⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩) [⟨.result 88369 .coefficient, false, none⟩])

def event88378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20395⟩⟩) (.product (.result 80012 .summary) (.transfer 88377) (⟨false, false, none, none, none⟩))

def event88379 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20395⟩⟩, .operator (⟨80012, 0⟩, ⟨88373, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩)

def event88380 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20393⟩⟩)

def event88381 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event88382 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event88383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event88384 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event88385 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event88386 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event88387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event88388 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event88389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 88388

def event88390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 88386

def event88391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 88389 .coefficient) (.value (.predecessor 1 88390 .coefficient)))

def event88392 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event88393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 88392

def event88394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 88384

def event88395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 88393 .coefficient, .predecessor 1 88394 .coefficient])

def event88396 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event88397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 88396

def event88398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 88382

def event88399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 88398 .coefficient))

def event88400 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event88401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 88400

def event88402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact88403RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88403RawTermsValid :
    exact88403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact88403RawTerms (.finite 2) 88402 .exactZero (none)

def event88404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 88400

def event88405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact88406RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact88406RawTermsValid :
    exact88406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact88406RawTerms (.finite 2) 88405 .exactZero (none)

def event88407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 88406

def event88408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 88403

def event88409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 88407 .coefficient) (.predecessor 1 88408 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩) [⟨.result 88406 .coefficient, true, some 1⟩, ⟨.result 88403 .coefficient, true, some 1⟩])

def event88411 : Event := .survivorFold (1) 88410

def exact88412RawTerms : List Term := []

theorem exact88412RawTermsValid :
    exact88412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88412 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact88412RawTerms (.finite 4) 88409 (.finite 4) (some (88410))

def event88413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 88412

def event88414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 88413 .coefficient))

def event88415 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event88416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14792⟩⟩) 0 ⟨10482⟩ 88415

def event88417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14792⟩⟩) (.authority (.programFamilyFact))

def exact88418RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact88418RawTermsValid :
    exact88418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14792⟩⟩) exact88418RawTerms (.finite 2) 88417 .exactZero (none)

def event88419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14793⟩⟩) 0 ⟨14792⟩ 88418

def event88420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.identity (.predecessor 0 88419 .coefficient))

def event88421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.finite 2)

def event88422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20392⟩⟩) 0 ⟨14793⟩ 88421

def event88423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20392⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact88424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩]

theorem exact88424RawTermsValid :
    exact88424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20392⟩⟩) exact88424RawTerms (.finite 136065468) 88423 .exactZero (none)

def event88425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact88426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact88426RawTermsValid :
    exact88426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88426 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact88426RawTerms .large 88425 .exactZero (none)

def event88427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20393⟩⟩) 0 ⟨6⟩ 88426

def event88428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20393⟩⟩) 1 ⟨20392⟩ 88424

def event88429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20393⟩⟩) (.product (.predecessor 0 88427 .coefficient) (.predecessor 1 88428 .coefficient) (⟨false, false, none, none, none⟩))

def event88430 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20393⟩⟩, .operator (⟨88426, 0⟩, ⟨88424, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩)

def exact88431RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩]

theorem exact88431RawTermsValid :
    exact88431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20393⟩⟩) exact88431RawTerms .large 88429 .exactZero (none)

def event88432 : Event := .preFoldPolynomial 88431 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩] .exactZero none

def exact88433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩, (1)⟩]

def event88433 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20393⟩⟩) 88432 exact88433RawTerms .large 88429 .exactZero (none)

def event88434 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26362⟩⟩)

def event88435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event88436 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event88437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event88438 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event88439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event88440 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event88441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event88442 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event88443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 88442

def event88444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 88440

def event88445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 88443 .coefficient) (.value (.predecessor 1 88444 .coefficient)))

def event88446 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event88447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 88446

def event88448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 88438

def event88449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 88447 .coefficient, .predecessor 1 88448 .coefficient])

def event88450 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event88451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 88450

def event88452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 88436

def event88453 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 88452 .coefficient))

def event88454 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event88455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 88454

def event88456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact88457RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88457RawTermsValid :
    exact88457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact88457RawTerms (.finite 2) 88456 .exactZero (none)

def event88458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 88454

def event88459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact88460RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact88460RawTermsValid :
    exact88460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88460 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact88460RawTerms (.finite 2) 88459 .exactZero (none)

def event88461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 88460

def event88462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 88457

def event88463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 88461 .coefficient) (.predecessor 1 88462 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88464 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10481⟩⟩, .operator (⟨88460, 0⟩, ⟨88457, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩)

def exact88465RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88465RawTermsValid :
    exact88465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88465 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact88465RawTerms (.finite 4) 88463 .exactZero (none)

def event88466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 88465

def event88467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 88466 .coefficient))

def event88468 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event88469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14792⟩⟩) 0 ⟨10482⟩ 88468

def event88470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14792⟩⟩) (.authority (.programFamilyFact))

def exact88471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact88471RawTermsValid :
    exact88471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14792⟩⟩) exact88471RawTerms (.finite 2) 88470 .exactZero (none)

def event88472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14793⟩⟩) 0 ⟨14792⟩ 88471

def event88473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.identity (.predecessor 0 88472 .coefficient))

def event88474 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.finite 2)

def event88475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23719⟩⟩) 0 ⟨14793⟩ 88474

def event88476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.authority (.programFamilyFact))

def event88477 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.finite 3720)

def event88478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event88479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23721⟩⟩) 0 ⟨6689⟩ 88478

def event88480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23721⟩⟩) 1 ⟨23719⟩ 88477

def event88481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23721⟩⟩) (.authority (.operator))

def exact88482RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩]

theorem exact88482RawTermsValid :
    exact88482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88482 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23721⟩⟩) exact88482RawTerms .large 88481 .exactZero (none)

def event88483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26358⟩⟩) 0 ⟨23721⟩ 88482

def event88484 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26358⟩⟩) (.authority (.operator))

def exact88485RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩]

theorem exact88485RawTermsValid :
    exact88485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26358⟩⟩) exact88485RawTerms (.finite 8192) 88484 .exactZero (none)

def event88486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event88487 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event88488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14832⟩⟩) 0 ⟨14793⟩ 88474

def event88489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14832⟩⟩) 1 ⟨110⟩ 88487

def event88490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14832⟩⟩) (.sum [.predecessor 0 88488 .coefficient, .predecessor 1 88489 .coefficient])

def event88491 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14832⟩⟩) (.finite 2)

def event88492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14833⟩⟩) 0 ⟨14832⟩ 88491

def event88493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14833⟩⟩) (.identity (.predecessor 0 88492 .coefficient))

def exact88494RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact88494RawTermsValid :
    exact88494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14833⟩⟩) exact88494RawTerms (.finite 2) 88493 .exactZero (none)

def event88495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact88496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88496RawTermsValid :
    exact88496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact88496RawTerms .large 88495 .exactZero (none)

def event88497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14834⟩⟩) 0 ⟨6544⟩ 88496

def event88498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14834⟩⟩) 1 ⟨14833⟩ 88494

def event88499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14834⟩⟩) (.product (.predecessor 0 88497 .coefficient) (.predecessor 1 88498 .coefficient) (⟨false, false, none, none, none⟩))

def event88500 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14834⟩⟩, .operator (⟨88496, 0⟩, ⟨88494, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88501RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88501RawTermsValid :
    exact88501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88501 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14834⟩⟩) exact88501RawTerms .large 88499 .exactZero (none)

def event88502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 88478

def event88503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact88504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact88504RawTermsValid :
    exact88504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact88504RawTerms .large 88503 .exactZero (none)

def event88505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14835⟩⟩) 0 ⟨6690⟩ 88504

def event88506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14835⟩⟩) 1 ⟨14834⟩ 88501

def event88507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14835⟩⟩) (.sum [.predecessor 0 88505 .coefficient, .predecessor 1 88506 .coefficient])

def exact88508RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88508RawTermsValid :
    exact88508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88508 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14835⟩⟩) exact88508RawTerms .large 88507 .exactZero (none)

def event88509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26359⟩⟩) 0 ⟨14835⟩ 88508

def event88510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26359⟩⟩) 1 ⟨26358⟩ 88485

def event88511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26359⟩⟩) (.product (.predecessor 0 88509 .coefficient) (.predecessor 1 88510 .coefficient) (⟨false, false, none, none, none⟩))

def event88512 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26359⟩⟩, .operator (⟨88508, 0⟩, ⟨88485, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩)

def event88513 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26359⟩⟩, .operator (⟨88508, 1⟩, ⟨88485, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩)

def event88514 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26359⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26358⟩⟩) ⟨23721⟩ 88482)

def event88515 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26359⟩⟩, .relation 88514 0, ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (-1)⟩)

def exact88516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (-1)⟩]

theorem exact88516RawTermsValid :
    exact88516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88516 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26359⟩⟩) exact88516RawTerms .large 88511 .exactZero (none)

def event88517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15265⟩⟩) 0 ⟨14793⟩ 88474

def event88518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15265⟩⟩) (.authority (.programFamilyFact))

def exact88519RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], []⟩, (1)⟩]

theorem exact88519RawTermsValid :
    exact88519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88519 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15265⟩⟩) exact88519RawTerms (.finite 43) 88518 .exactZero (none)

def event88520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15266⟩⟩) 0 ⟨6544⟩ 88496

def event88521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15266⟩⟩) 1 ⟨15265⟩ 88519

def event88522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15266⟩⟩) (.product (.predecessor 0 88520 .coefficient) (.predecessor 1 88521 .coefficient) (⟨false, true, none, none, some 1⟩))

def event88523 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15266⟩⟩, .operator (⟨88496, 0⟩, ⟨88519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88524RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88524RawTermsValid :
    exact88524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88524 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15266⟩⟩) exact88524RawTerms .large 88522 .exactZero (none)

def event88525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 88478

def event88526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact88527RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact88527RawTermsValid :
    exact88527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88527 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact88527RawTerms .large 88526 .exactZero (none)

def event88528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15267⟩⟩) 0 ⟨6709⟩ 88527

def event88529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15267⟩⟩) 1 ⟨15266⟩ 88524

def event88530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15267⟩⟩) (.sum [.predecessor 0 88528 .coefficient, .predecessor 1 88529 .coefficient])

def exact88531RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88531RawTermsValid :
    exact88531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15267⟩⟩) exact88531RawTerms .large 88530 .exactZero (none)

def event88532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26362⟩⟩) 0 ⟨15267⟩ 88531

def event88533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26362⟩⟩) 1 ⟨26359⟩ 88516

def event88534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26362⟩⟩) (.sum [.predecessor 0 88532 .coefficient, .predecessor 1 88533 .coefficient])

def exact88535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88535RawTermsValid :
    exact88535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26362⟩⟩) exact88535RawTerms .large 88534 .exactZero (none)

def event88536 : Event := .preFoldPolynomial 88535 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact88537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event88537 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26362⟩⟩) 88536 exact88537RawTerms .large 88534 .exactZero (none)

def event88538 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14793⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨88380, 88538⟩

def event88539 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20395⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩) (1) 0 2 (.universal 88538 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩) (none) 88537)

def event88540 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20395⟩⟩, .relation 88539 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def event88541 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20395⟩⟩, .relation 88539 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩)

def event88542 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20395⟩⟩, .relation 88539 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩)

def event88543 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20395⟩⟩, .relation 88539 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact88544RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88544RawTermsValid :
    exact88544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20395⟩⟩) exact88544RawTerms .large 88376 (.finite 1811303510016) (some (88378))

def event88545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26361⟩⟩) 0 ⟨20395⟩ 88544

def event88546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26361⟩⟩) 1 ⟨26360⟩ 88366

def event88547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26361⟩⟩) (.sum [.predecessor 0 88545 .coefficient, .predecessor 1 88546 .coefficient])

def event88548 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26361⟩⟩, .operator (⟨88544, 0⟩, ⟨88366, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩)

def event88549 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26361⟩⟩, .operator (⟨88544, 2⟩, ⟨88366, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (-1)⟩)

def event88550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26361⟩⟩) (.sum [.result 88544 .summary, .result 88366 .summary])

def exact88551RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88551RawTermsValid :
    exact88551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88551 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26361⟩⟩) exact88551RawTerms .large 88547 (.finite 1291889174379421642752) (some (88550))

def event88552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26568⟩⟩) 0 ⟨26361⟩ 88551

def event88553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26568⟩⟩) 1 ⟨26567⟩ 88071

def event88554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26568⟩⟩) (.sum [.predecessor 0 88552 .coefficient, .predecessor 1 88553 .coefficient])

def event88555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26568⟩⟩) (.sum [.result 88551 .summary, .result 88071 .summary])

def exact88556RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88556RawTermsValid :
    exact88556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88556 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26568⟩⟩) exact88556RawTerms .large 88554 (.finite 2583789554981353578496) (some (88555))

def event88557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26785⟩⟩) 0 ⟨26568⟩ 88556

def event88558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26785⟩⟩) 1 ⟨26784⟩ 87591

def event88559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26785⟩⟩) (.sum [.predecessor 0 88557 .coefficient, .predecessor 1 88558 .coefficient])

def event88560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26785⟩⟩) (.sum [.result 88556 .summary, .result 87591 .summary])

def exact88561RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88561RawTermsValid :
    exact88561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26785⟩⟩) exact88561RawTerms .large 88559 (.finite 3875701141805795807232) (some (88560))

def event88562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27002⟩⟩) 0 ⟨26785⟩ 88561

def event88563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27002⟩⟩) 1 ⟨27001⟩ 87111

def event88564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27002⟩⟩) (.sum [.predecessor 0 88562 .coefficient, .predecessor 1 88563 .coefficient])

def event88565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27002⟩⟩) (.sum [.result 88561 .summary, .result 87111 .summary])

def exact88566RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88566RawTermsValid :
    exact88566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88566 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27002⟩⟩) exact88566RawTerms .large 88564 (.finite 5167635141075258621952) (some (88565))

def event88567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27219⟩⟩) 0 ⟨27002⟩ 88566

def event88568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27219⟩⟩) 1 ⟨27218⟩ 86631

def event88569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27219⟩⟩) (.sum [.predecessor 0 88567 .coefficient, .predecessor 1 88568 .coefficient])

def event88570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27219⟩⟩) (.sum [.result 88566 .summary, .result 86631 .summary])

def exact88571RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88571RawTermsValid :
    exact88571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88571 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27219⟩⟩) exact88571RawTerms .large 88569 (.finite 6459613965234762608640) (some (88570))

def event88572 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27436⟩⟩) 0 ⟨27219⟩ 88571

def event88573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27436⟩⟩) 1 ⟨27435⟩ 86151

def event88574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27436⟩⟩) (.sum [.predecessor 0 88572 .coefficient, .predecessor 1 88573 .coefficient])

def event88575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27436⟩⟩) (.sum [.result 88571 .summary, .result 86151 .summary])

def eventLeaf5520 : Array AnnotatedEvent := #[
  { event := event88320
    frameStart := 88227 },
  { event := event88321
    frameStart := 88227 },
  { event := event88322
    frameStart := 88227 },
  { event := event88323
    frameStart := 88227 },
  { event := event88324
    frameStart := 88227 },
  { event := event88325
    frameStart := 88227 },
  { event := event88326
    frameStart := 88227 },
  { event := event88327
    frameStart := 88227 },
  { event := event88328
    frameStart := 88227 },
  { event := event88329
    frameStart := 88227 },
  { event := event88330
    frameStart := 88227 },
  { event := event88331
    frameStart := 88227 },
  { event := event88332
    frameStart := 88227 },
  { event := event88333
    frameStart := 88227 },
  { event := event88334
    frameStart := 88227 },
  { event := event88335
    frameStart := 88227 }
]

def eventLeaf5521 : Array AnnotatedEvent := #[
  { event := event88336
    frameStart := 88227 },
  { event := event88337
    frameStart := 88227 },
  { event := event88338
    frameStart := 88227 },
  { event := event88339
    frameStart := 88227 },
  { event := event88340
    frameStart := 88227 },
  { event := event88341
    frameStart := 88227 },
  { event := event88342
    frameStart := 88227 },
  { event := event88343
    frameStart := 0 },
  { event := event88344
    frameStart := 0 },
  { event := event88345
    frameStart := 0 },
  { event := event88346
    frameStart := 0 },
  { event := event88347
    frameStart := 0 },
  { event := event88348
    frameStart := 0 },
  { event := event88349
    frameStart := 0 },
  { event := event88350
    frameStart := 0 },
  { event := event88351
    frameStart := 0 }
]

def eventLeaf5522 : Array AnnotatedEvent := #[
  { event := event88352
    frameStart := 0 },
  { event := event88353
    frameStart := 0 },
  { event := event88354
    frameStart := 0 },
  { event := event88355
    frameStart := 0 },
  { event := event88356
    frameStart := 0 },
  { event := event88357
    frameStart := 0 },
  { event := event88358
    frameStart := 0 },
  { event := event88359
    frameStart := 0 },
  { event := event88360
    frameStart := 0 },
  { event := event88361
    frameStart := 0 },
  { event := event88362
    frameStart := 0 },
  { event := event88363
    frameStart := 0 },
  { event := event88364
    frameStart := 0 },
  { event := event88365
    frameStart := 0 },
  { event := event88366
    frameStart := 0 },
  { event := event88367
    frameStart := 0 }
]

def eventLeaf5523 : Array AnnotatedEvent := #[
  { event := event88368
    frameStart := 0 },
  { event := event88369
    frameStart := 0 },
  { event := event88370
    frameStart := 0 },
  { event := event88371
    frameStart := 0 },
  { event := event88372
    frameStart := 0 },
  { event := event88373
    frameStart := 0 },
  { event := event88374
    frameStart := 0 },
  { event := event88375
    frameStart := 0 },
  { event := event88376
    frameStart := 0 },
  { event := event88377
    frameStart := 0 },
  { event := event88378
    frameStart := 0 },
  { event := event88379
    frameStart := 0 },
  { event := event88380
    frameStart := 88380 },
  { event := event88381
    frameStart := 88380 },
  { event := event88382
    frameStart := 88380 },
  { event := event88383
    frameStart := 88380 }
]

def eventLeaf5524 : Array AnnotatedEvent := #[
  { event := event88384
    frameStart := 88380 },
  { event := event88385
    frameStart := 88380 },
  { event := event88386
    frameStart := 88380 },
  { event := event88387
    frameStart := 88380 },
  { event := event88388
    frameStart := 88380 },
  { event := event88389
    frameStart := 88380 },
  { event := event88390
    frameStart := 88380 },
  { event := event88391
    frameStart := 88380 },
  { event := event88392
    frameStart := 88380 },
  { event := event88393
    frameStart := 88380 },
  { event := event88394
    frameStart := 88380 },
  { event := event88395
    frameStart := 88380 },
  { event := event88396
    frameStart := 88380 },
  { event := event88397
    frameStart := 88380 },
  { event := event88398
    frameStart := 88380 },
  { event := event88399
    frameStart := 88380 }
]

def eventLeaf5525 : Array AnnotatedEvent := #[
  { event := event88400
    frameStart := 88380 },
  { event := event88401
    frameStart := 88380 },
  { event := event88402
    frameStart := 88380 },
  { event := event88403
    frameStart := 88380 },
  { event := event88404
    frameStart := 88380 },
  { event := event88405
    frameStart := 88380 },
  { event := event88406
    frameStart := 88380 },
  { event := event88407
    frameStart := 88380 },
  { event := event88408
    frameStart := 88380 },
  { event := event88409
    frameStart := 88380 },
  { event := event88410
    frameStart := 88380 },
  { event := event88411
    frameStart := 88380 },
  { event := event88412
    frameStart := 88380 },
  { event := event88413
    frameStart := 88380 },
  { event := event88414
    frameStart := 88380 },
  { event := event88415
    frameStart := 88380 }
]

def eventLeaf5526 : Array AnnotatedEvent := #[
  { event := event88416
    frameStart := 88380 },
  { event := event88417
    frameStart := 88380 },
  { event := event88418
    frameStart := 88380 },
  { event := event88419
    frameStart := 88380 },
  { event := event88420
    frameStart := 88380 },
  { event := event88421
    frameStart := 88380 },
  { event := event88422
    frameStart := 88380 },
  { event := event88423
    frameStart := 88380 },
  { event := event88424
    frameStart := 88380 },
  { event := event88425
    frameStart := 88380 },
  { event := event88426
    frameStart := 88380 },
  { event := event88427
    frameStart := 88380 },
  { event := event88428
    frameStart := 88380 },
  { event := event88429
    frameStart := 88380 },
  { event := event88430
    frameStart := 88380 },
  { event := event88431
    frameStart := 88380 }
]

def eventLeaf5527 : Array AnnotatedEvent := #[
  { event := event88432
    frameStart := 88380 },
  { event := event88433
    frameStart := 88380 },
  { event := event88434
    frameStart := 88434 },
  { event := event88435
    frameStart := 88434 },
  { event := event88436
    frameStart := 88434 },
  { event := event88437
    frameStart := 88434 },
  { event := event88438
    frameStart := 88434 },
  { event := event88439
    frameStart := 88434 },
  { event := event88440
    frameStart := 88434 },
  { event := event88441
    frameStart := 88434 },
  { event := event88442
    frameStart := 88434 },
  { event := event88443
    frameStart := 88434 },
  { event := event88444
    frameStart := 88434 },
  { event := event88445
    frameStart := 88434 },
  { event := event88446
    frameStart := 88434 },
  { event := event88447
    frameStart := 88434 }
]

def eventLeaf5528 : Array AnnotatedEvent := #[
  { event := event88448
    frameStart := 88434 },
  { event := event88449
    frameStart := 88434 },
  { event := event88450
    frameStart := 88434 },
  { event := event88451
    frameStart := 88434 },
  { event := event88452
    frameStart := 88434 },
  { event := event88453
    frameStart := 88434 },
  { event := event88454
    frameStart := 88434 },
  { event := event88455
    frameStart := 88434 },
  { event := event88456
    frameStart := 88434 },
  { event := event88457
    frameStart := 88434 },
  { event := event88458
    frameStart := 88434 },
  { event := event88459
    frameStart := 88434 },
  { event := event88460
    frameStart := 88434 },
  { event := event88461
    frameStart := 88434 },
  { event := event88462
    frameStart := 88434 },
  { event := event88463
    frameStart := 88434 }
]

def eventLeaf5529 : Array AnnotatedEvent := #[
  { event := event88464
    frameStart := 88434 },
  { event := event88465
    frameStart := 88434 },
  { event := event88466
    frameStart := 88434 },
  { event := event88467
    frameStart := 88434 },
  { event := event88468
    frameStart := 88434 },
  { event := event88469
    frameStart := 88434 },
  { event := event88470
    frameStart := 88434 },
  { event := event88471
    frameStart := 88434 },
  { event := event88472
    frameStart := 88434 },
  { event := event88473
    frameStart := 88434 },
  { event := event88474
    frameStart := 88434 },
  { event := event88475
    frameStart := 88434 },
  { event := event88476
    frameStart := 88434 },
  { event := event88477
    frameStart := 88434 },
  { event := event88478
    frameStart := 88434 },
  { event := event88479
    frameStart := 88434 }
]

def eventLeaf5530 : Array AnnotatedEvent := #[
  { event := event88480
    frameStart := 88434 },
  { event := event88481
    frameStart := 88434 },
  { event := event88482
    frameStart := 88434 },
  { event := event88483
    frameStart := 88434 },
  { event := event88484
    frameStart := 88434 },
  { event := event88485
    frameStart := 88434 },
  { event := event88486
    frameStart := 88434 },
  { event := event88487
    frameStart := 88434 },
  { event := event88488
    frameStart := 88434 },
  { event := event88489
    frameStart := 88434 },
  { event := event88490
    frameStart := 88434 },
  { event := event88491
    frameStart := 88434 },
  { event := event88492
    frameStart := 88434 },
  { event := event88493
    frameStart := 88434 },
  { event := event88494
    frameStart := 88434 },
  { event := event88495
    frameStart := 88434 }
]

def eventLeaf5531 : Array AnnotatedEvent := #[
  { event := event88496
    frameStart := 88434 },
  { event := event88497
    frameStart := 88434 },
  { event := event88498
    frameStart := 88434 },
  { event := event88499
    frameStart := 88434 },
  { event := event88500
    frameStart := 88434 },
  { event := event88501
    frameStart := 88434 },
  { event := event88502
    frameStart := 88434 },
  { event := event88503
    frameStart := 88434 },
  { event := event88504
    frameStart := 88434 },
  { event := event88505
    frameStart := 88434 },
  { event := event88506
    frameStart := 88434 },
  { event := event88507
    frameStart := 88434 },
  { event := event88508
    frameStart := 88434 },
  { event := event88509
    frameStart := 88434 },
  { event := event88510
    frameStart := 88434 },
  { event := event88511
    frameStart := 88434 }
]

def eventLeaf5532 : Array AnnotatedEvent := #[
  { event := event88512
    frameStart := 88434 },
  { event := event88513
    frameStart := 88434 },
  { event := event88514
    frameStart := 88434 },
  { event := event88515
    frameStart := 88434 },
  { event := event88516
    frameStart := 88434 },
  { event := event88517
    frameStart := 88434 },
  { event := event88518
    frameStart := 88434 },
  { event := event88519
    frameStart := 88434 },
  { event := event88520
    frameStart := 88434 },
  { event := event88521
    frameStart := 88434 },
  { event := event88522
    frameStart := 88434 },
  { event := event88523
    frameStart := 88434 },
  { event := event88524
    frameStart := 88434 },
  { event := event88525
    frameStart := 88434 },
  { event := event88526
    frameStart := 88434 },
  { event := event88527
    frameStart := 88434 }
]

def eventLeaf5533 : Array AnnotatedEvent := #[
  { event := event88528
    frameStart := 88434 },
  { event := event88529
    frameStart := 88434 },
  { event := event88530
    frameStart := 88434 },
  { event := event88531
    frameStart := 88434 },
  { event := event88532
    frameStart := 88434 },
  { event := event88533
    frameStart := 88434 },
  { event := event88534
    frameStart := 88434 },
  { event := event88535
    frameStart := 88434 },
  { event := event88536
    frameStart := 88434 },
  { event := event88537
    frameStart := 88434 },
  { event := event88538
    frameStart := 0 },
  { event := event88539
    frameStart := 0 },
  { event := event88540
    frameStart := 0 },
  { event := event88541
    frameStart := 0 },
  { event := event88542
    frameStart := 0 },
  { event := event88543
    frameStart := 0 }
]

def eventLeaf5534 : Array AnnotatedEvent := #[
  { event := event88544
    frameStart := 0 },
  { event := event88545
    frameStart := 0 },
  { event := event88546
    frameStart := 0 },
  { event := event88547
    frameStart := 0 },
  { event := event88548
    frameStart := 0 },
  { event := event88549
    frameStart := 0 },
  { event := event88550
    frameStart := 0 },
  { event := event88551
    frameStart := 0 },
  { event := event88552
    frameStart := 0 },
  { event := event88553
    frameStart := 0 },
  { event := event88554
    frameStart := 0 },
  { event := event88555
    frameStart := 0 },
  { event := event88556
    frameStart := 0 },
  { event := event88557
    frameStart := 0 },
  { event := event88558
    frameStart := 0 },
  { event := event88559
    frameStart := 0 }
]

def eventLeaf5535 : Array AnnotatedEvent := #[
  { event := event88560
    frameStart := 0 },
  { event := event88561
    frameStart := 0 },
  { event := event88562
    frameStart := 0 },
  { event := event88563
    frameStart := 0 },
  { event := event88564
    frameStart := 0 },
  { event := event88565
    frameStart := 0 },
  { event := event88566
    frameStart := 0 },
  { event := event88567
    frameStart := 0 },
  { event := event88568
    frameStart := 0 },
  { event := event88569
    frameStart := 0 },
  { event := event88570
    frameStart := 0 },
  { event := event88571
    frameStart := 0 },
  { event := event88572
    frameStart := 0 },
  { event := event88573
    frameStart := 0 },
  { event := event88574
    frameStart := 0 },
  { event := event88575
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events345
