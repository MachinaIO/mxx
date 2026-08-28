import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events380

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event97280 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16372⟩⟩) (.identity (.predecessor 0 97279 .coefficient))

def event97281 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16372⟩⟩) (.finite 36)

def event97282 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21965⟩⟩) 0 ⟨16372⟩ 97281

def event97283 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21965⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact97284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩, (1)⟩]

theorem exact97284RawTermsValid :
    exact97284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97284 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21965⟩⟩) exact97284RawTerms (.finite 136065468) 97283 .exactZero (none)

def event97285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact97286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact97286RawTermsValid :
    exact97286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97286 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact97286RawTerms .large 97285 .exactZero (none)

def event97287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21966⟩⟩) 0 ⟨6⟩ 97286

def event97288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21966⟩⟩) 1 ⟨21965⟩ 97284

def event97289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21966⟩⟩) (.product (.predecessor 0 97287 .coefficient) (.predecessor 1 97288 .coefficient) (⟨false, false, none, none, none⟩))

def event97290 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21966⟩⟩, .operator (⟨97286, 0⟩, ⟨97284, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩, (1)⟩)

def exact97291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩, (1)⟩]

theorem exact97291RawTermsValid :
    exact97291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21966⟩⟩) exact97291RawTerms .large 97289 .exactZero (none)

def event97292 : Event := .preFoldPolynomial 97291 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩, (1)⟩] .exactZero none

def exact97293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩, (1)⟩]

def event97293 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21966⟩⟩) 97292 exact97293RawTerms .large 97289 .exactZero (none)

def event97294 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28704⟩⟩)

def event97295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97296 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97298 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97298

def event97300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97296

def event97301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97299 .coefficient) (.value (.predecessor 1 97300 .coefficient)))

def event97302 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11933⟩⟩) 0 ⟨5503⟩ 97302

def event97304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11933⟩⟩) (.authority (.programFamilyFact))

def exact97305RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11933⟩⟩], []⟩, (1)⟩]

theorem exact97305RawTermsValid :
    exact97305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11933⟩⟩) exact97305RawTerms (.finite 36) 97304 .exactZero (none)

def event97306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9700⟩⟩) 0 ⟨5503⟩ 97302

def event97307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9700⟩⟩) (.authority (.programFamilyFact))

def exact97308RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩], []⟩, (1)⟩]

theorem exact97308RawTermsValid :
    exact97308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97308 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9700⟩⟩) exact97308RawTerms (.finite 36) 97307 .exactZero (none)

def event97309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11934⟩⟩) 0 ⟨9700⟩ 97308

def event97310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11934⟩⟩) 1 ⟨11933⟩ 97305

def event97311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11934⟩⟩) (.product (.predecessor 0 97309 .coefficient) (.predecessor 1 97310 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97312 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11934⟩⟩, .operator (⟨97308, 0⟩, ⟨97305, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], []⟩, (1)⟩)

def exact97313RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], []⟩, (1)⟩]

theorem exact97313RawTermsValid :
    exact97313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97313 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11934⟩⟩) exact97313RawTerms (.finite 1296) 97311 .exactZero (none)

def event97314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11935⟩⟩) 0 ⟨11934⟩ 97313

def event97315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11935⟩⟩) (.identity (.predecessor 0 97314 .coefficient))

def event97316 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11935⟩⟩) (.finite 1296)

def event97317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16371⟩⟩) 0 ⟨11935⟩ 97316

def event97318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16371⟩⟩) (.authority (.programFamilyFact))

def exact97319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], []⟩, (1)⟩]

theorem exact97319RawTermsValid :
    exact97319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16371⟩⟩) exact97319RawTerms (.finite 36) 97318 .exactZero (none)

def event97320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16372⟩⟩) 0 ⟨16371⟩ 97319

def event97321 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16372⟩⟩) (.identity (.predecessor 0 97320 .coefficient))

def event97322 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16372⟩⟩) (.finite 36)

def event97323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24403⟩⟩) 0 ⟨16372⟩ 97322

def event97324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24403⟩⟩) (.authority (.programFamilyFact))

def event97325 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24403⟩⟩) (.finite 3720)

def event97326 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event97327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24405⟩⟩) 0 ⟨6689⟩ 97326

def event97328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24405⟩⟩) 1 ⟨24403⟩ 97325

def event97329 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24405⟩⟩) (.authority (.operator))

def exact97330RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩]

theorem exact97330RawTermsValid :
    exact97330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97330 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24405⟩⟩) exact97330RawTerms .large 97329 .exactZero (none)

def event97331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28699⟩⟩) 0 ⟨24405⟩ 97330

def event97332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28699⟩⟩) (.authority (.operator))

def exact97333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (1)⟩]

theorem exact97333RawTermsValid :
    exact97333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28699⟩⟩) exact97333RawTerms (.finite 8192) 97332 .exactZero (none)

def event97334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event97335 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event97336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16413⟩⟩) 0 ⟨16372⟩ 97322

def event97337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16413⟩⟩) 1 ⟨110⟩ 97335

def event97338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16413⟩⟩) (.sum [.predecessor 0 97336 .coefficient, .predecessor 1 97337 .coefficient])

def event97339 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16413⟩⟩) (.finite 36)

def event97340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16414⟩⟩) 0 ⟨16413⟩ 97339

def event97341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16414⟩⟩) (.identity (.predecessor 0 97340 .coefficient))

def exact97342RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], []⟩, (1)⟩]

theorem exact97342RawTermsValid :
    exact97342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97342 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16414⟩⟩) exact97342RawTerms (.finite 36) 97341 .exactZero (none)

def event97343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact97344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97344RawTermsValid :
    exact97344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact97344RawTerms .large 97343 .exactZero (none)

def event97345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16415⟩⟩) 0 ⟨6544⟩ 97344

def event97346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16415⟩⟩) 1 ⟨16414⟩ 97342

def event97347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16415⟩⟩) (.product (.predecessor 0 97345 .coefficient) (.predecessor 1 97346 .coefficient) (⟨false, false, none, none, none⟩))

def event97348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16415⟩⟩, .operator (⟨97344, 0⟩, ⟨97342, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97349RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97349RawTermsValid :
    exact97349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97349 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16415⟩⟩) exact97349RawTerms .large 97347 .exactZero (none)

def event97350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 97326

def event97351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact97352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact97352RawTermsValid :
    exact97352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97352 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact97352RawTerms .large 97351 .exactZero (none)

def event97353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16416⟩⟩) 0 ⟨6701⟩ 97352

def event97354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16416⟩⟩) 1 ⟨16415⟩ 97349

def event97355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16416⟩⟩) (.sum [.predecessor 0 97353 .coefficient, .predecessor 1 97354 .coefficient])

def exact97356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97356RawTermsValid :
    exact97356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97356 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16416⟩⟩) exact97356RawTerms .large 97355 .exactZero (none)

def event97357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28700⟩⟩) 0 ⟨16416⟩ 97356

def event97358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28700⟩⟩) 1 ⟨28699⟩ 97333

def event97359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28700⟩⟩) (.product (.predecessor 0 97357 .coefficient) (.predecessor 1 97358 .coefficient) (⟨false, false, none, none, none⟩))

def event97360 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28700⟩⟩, .operator (⟨97356, 0⟩, ⟨97333, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (1)⟩)

def event97361 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28700⟩⟩, .operator (⟨97356, 1⟩, ⟨97333, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩)

def event97362 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28700⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28699⟩⟩) ⟨24405⟩ 97330)

def event97363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28700⟩⟩, .relation 97362 0, ⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (-1)⟩)

def exact97364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (-1)⟩]

theorem exact97364RawTermsValid :
    exact97364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28700⟩⟩) exact97364RawTerms .large 97359 .exactZero (none)

def event97365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17113⟩⟩) 0 ⟨16372⟩ 97322

def event97366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17113⟩⟩) (.authority (.programFamilyFact))

def exact97367RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], []⟩, (1)⟩]

theorem exact97367RawTermsValid :
    exact97367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17113⟩⟩) exact97367RawTerms (.finite 62) 97366 .exactZero (none)

def event97368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17114⟩⟩) 0 ⟨6544⟩ 97344

def event97369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17114⟩⟩) 1 ⟨17113⟩ 97367

def event97370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17114⟩⟩) (.product (.predecessor 0 97368 .coefficient) (.predecessor 1 97369 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97371 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17114⟩⟩, .operator (⟨97344, 0⟩, ⟨97367, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97372RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97372RawTermsValid :
    exact97372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17114⟩⟩) exact97372RawTerms .large 97370 .exactZero (none)

def event97373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 97326

def event97374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6731⟩⟩) (.authority (.operator))

def exact97375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact97375RawTermsValid :
    exact97375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97375 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6731⟩⟩) exact97375RawTerms .large 97374 .exactZero (none)

def event97376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17115⟩⟩) 0 ⟨6731⟩ 97375

def event97377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17115⟩⟩) 1 ⟨17114⟩ 97372

def event97378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17115⟩⟩) (.sum [.predecessor 0 97376 .coefficient, .predecessor 1 97377 .coefficient])

def exact97379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97379RawTermsValid :
    exact97379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17115⟩⟩) exact97379RawTerms .large 97378 .exactZero (none)

def event97380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28704⟩⟩) 0 ⟨17115⟩ 97379

def event97381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28704⟩⟩) 1 ⟨28700⟩ 97364

def event97382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28704⟩⟩) (.sum [.predecessor 0 97380 .coefficient, .predecessor 1 97381 .coefficient])

def exact97383RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97383RawTermsValid :
    exact97383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97383 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28704⟩⟩) exact97383RawTerms .large 97382 .exactZero (none)

def event97384 : Event := .preFoldPolynomial 97383 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact97385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event97385 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28704⟩⟩) 97384 exact97385RawTerms .large 97382 .exactZero (none)

def event97386 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16372⟩⟩) ⟨⟨144⟩, ⟨52⟩, ⟨109⟩⟩ ⟨97252, 97386⟩

def event97387 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21968⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩) (1) 0 2 (.universal 97386 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21965⟩⟩]⟩) (none) 97385)

def event97388 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21968⟩⟩, .relation 97387 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩)

def event97389 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21968⟩⟩, .relation 97387 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩)

def event97390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21968⟩⟩, .relation 97387 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩)

def event97391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21968⟩⟩, .relation 97387 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact97392RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97392RawTermsValid :
    exact97392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97392 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21968⟩⟩) exact97392RawTerms .large 97248 (.finite 1811303510016) (some (97250))

def event97393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28702⟩⟩) 0 ⟨21968⟩ 97392

def event97394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28702⟩⟩) 1 ⟨28701⟩ 97238

def event97395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28702⟩⟩) (.sum [.predecessor 0 97393 .coefficient, .predecessor 1 97394 .coefficient])

def event97396 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28702⟩⟩, .operator (⟨97392, 0⟩, ⟨97238, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28699⟩⟩]⟩, (1)⟩)

def event97397 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28702⟩⟩, .operator (⟨97392, 2⟩, ⟨97238, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨24405⟩⟩]⟩, (-1)⟩)

def event97398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28702⟩⟩) (.sum [.result 97392 .summary, .result 97238 .summary])

def exact97399RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97399RawTermsValid :
    exact97399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97399 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28702⟩⟩) exact97399RawTerms .large 97395 (.finite 1292270185944771604480) (some (97398))

def event97400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24340⟩⟩) 0 ⟨16253⟩ 4744

def event97401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24340⟩⟩) (.authority (.programFamilyFact))

def event97402 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24340⟩⟩) (.finite 3720)

def event97403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24342⟩⟩) 0 ⟨6689⟩ 5477

def event97404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24342⟩⟩) 1 ⟨24340⟩ 97402

def event97405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24342⟩⟩) (.authority (.operator))

def exact97406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩]

theorem exact97406RawTermsValid :
    exact97406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24342⟩⟩) exact97406RawTerms .large 97405 .exactZero (none)

def event97407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28482⟩⟩) 0 ⟨24342⟩ 97406

def event97408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28482⟩⟩) (.authority (.operator))

def exact97409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩]

theorem exact97409RawTermsValid :
    exact97409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28482⟩⟩) exact97409RawTerms (.finite 8192) 97408 .exactZero (none)

def event97410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23073⟩⟩) 0 ⟨11739⟩ 4738

def event97411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23073⟩⟩) (.authority (.programFamilyFact))

def event97412 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23073⟩⟩) (.finite 3720)

def event97413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23074⟩⟩) 0 ⟨6689⟩ 5477

def event97414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23074⟩⟩) 1 ⟨23073⟩ 97412

def event97415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23074⟩⟩) (.authority (.operator))

def exact97416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (1)⟩]

theorem exact97416RawTermsValid :
    exact97416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23074⟩⟩) exact97416RawTerms .large 97415 .exactZero (none)

def event97417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25129⟩⟩) 0 ⟨23074⟩ 97416

def event97418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25129⟩⟩) (.authority (.operator))

def exact97419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩]

theorem exact97419RawTermsValid :
    exact97419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25129⟩⟩) exact97419RawTerms (.finite 8192) 97418 .exactZero (none)

def event97420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11740⟩⟩) 0 ⟨11737⟩ 4727

def event97421 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11740⟩⟩) 1 ⟨6564⟩ 32

def event97422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11740⟩⟩) (.tensor (.predecessor 0 97420 .coefficient) (.predecessor 1 97421 .coefficient) true false)

def event97423 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11740⟩⟩, .operator (⟨4727, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97424RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97424RawTermsValid :
    exact97424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11740⟩⟩) exact97424RawTerms .large 97422 .exactZero (none)

def event97425 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7120⟩⟩) 0 ⟨5506⟩ 27

def event97426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7120⟩⟩) 1 ⟨6783⟩ 9979

def event97427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7120⟩⟩) (.product (.predecessor 0 97425 .coefficient) (.predecessor 1 97426 .coefficient) (⟨false, false, none, none, none⟩))

def event97428 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7120⟩⟩, .operator (⟨27, 0⟩, ⟨9979, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact97429RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact97429RawTermsValid :
    exact97429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7120⟩⟩) exact97429RawTerms .large 97427 .exactZero (none)

def event97430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11741⟩⟩) 0 ⟨7120⟩ 97429

def event97431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11741⟩⟩) 1 ⟨11740⟩ 97424

def event97432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11741⟩⟩) (.sum [.predecessor 0 97430 .coefficient, .predecessor 1 97431 .coefficient])

def exact97433RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97433RawTermsValid :
    exact97433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97433 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11741⟩⟩) exact97433RawTerms .large 97432 .exactZero (none)

def event97434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11742⟩⟩) 0 ⟨11741⟩ 97433

def event97435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11742⟩⟩) 1 ⟨97⟩ 9971

def event97436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11742⟩⟩) (.sum [.predecessor 0 97434 .coefficient, .predecessor 1 97435 .coefficient])

def event97437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11742⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) [⟨.result 9971 .coefficient, false, none⟩])

def event97438 : Event := .survivorFold (1) 97437

def exact97439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97439RawTermsValid :
    exact97439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11742⟩⟩) exact97439RawTerms .large 97436 (.finite 26) (some (97437))

def event97440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11743⟩⟩) 0 ⟨11742⟩ 97439

def event97441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11743⟩⟩) 1 ⟨9595⟩ 4730

def event97442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11743⟩⟩) (.product (.predecessor 0 97440 .coefficient) (.predecessor 1 97441 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97443 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11743⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩) [⟨.result 4730 .coefficient, true, some 1⟩])

def event97444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11743⟩⟩) (.product (.result 97439 .summary) (.transfer 97443) (⟨false, false, none, none, none⟩))

def event97445 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11743⟩⟩, .operator (⟨97439, 1⟩, ⟨4730, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event97446 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11743⟩⟩, .operator (⟨97439, 0⟩, ⟨4730, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact97447RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97447RawTermsValid :
    exact97447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97447 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11743⟩⟩) exact97447RawTerms .large 97442 (.finite 24960) (some (97444))

def event97448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9596⟩⟩) 0 ⟨9595⟩ 4730

def event97449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9596⟩⟩) 1 ⟨6564⟩ 32

def event97450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9596⟩⟩) (.tensor (.predecessor 0 97448 .coefficient) (.predecessor 1 97449 .coefficient) true false)

def event97451 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9596⟩⟩, .operator (⟨4730, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97452RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97452RawTermsValid :
    exact97452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97452 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9596⟩⟩) exact97452RawTerms .large 97450 .exactZero (none)

def event97453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7100⟩⟩) 0 ⟨5506⟩ 27

def event97454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7100⟩⟩) 1 ⟨6763⟩ 10020

def event97455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7100⟩⟩) (.product (.predecessor 0 97453 .coefficient) (.predecessor 1 97454 .coefficient) (⟨false, false, none, none, none⟩))

def event97456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7100⟩⟩, .operator (⟨27, 0⟩, ⟨10020, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩)

def exact97457RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact97457RawTermsValid :
    exact97457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7100⟩⟩) exact97457RawTerms .large 97455 .exactZero (none)

def event97458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9597⟩⟩) 0 ⟨7100⟩ 97457

def event97459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9597⟩⟩) 1 ⟨9596⟩ 97452

def event97460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9597⟩⟩) (.sum [.predecessor 0 97458 .coefficient, .predecessor 1 97459 .coefficient])

def exact97461RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97461RawTermsValid :
    exact97461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9597⟩⟩) exact97461RawTerms .large 97460 .exactZero (none)

def event97462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9598⟩⟩) 0 ⟨9597⟩ 97461

def event97463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9598⟩⟩) 1 ⟨77⟩ 10012

def event97464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9598⟩⟩) (.sum [.predecessor 0 97462 .coefficient, .predecessor 1 97463 .coefficient])

def event97465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9598⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) [⟨.result 10012 .coefficient, false, none⟩])

def event97466 : Event := .survivorFold (1) 97465

def exact97467RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97467RawTermsValid :
    exact97467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97467 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9598⟩⟩) exact97467RawTerms .large 97464 (.finite 26) (some (97465))

def event97468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9599⟩⟩) 0 ⟨9598⟩ 97467

def event97469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9599⟩⟩) 1 ⟨7862⟩ 10009

def event97470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9599⟩⟩) (.product (.predecessor 0 97468 .coefficient) (.predecessor 1 97469 .coefficient) (⟨false, false, none, none, none⟩))

def event97471 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9599⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) [⟨.result 10005 .coefficient, false, none⟩])

def event97472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9599⟩⟩) (.product (.result 97467 .summary) (.transfer 97471) (⟨false, false, none, none, none⟩))

def event97473 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9599⟩⟩, .operator (⟨97467, 1⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (-1)⟩)

def event97474 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9599⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7861⟩⟩) ⟨6783⟩ 9979)

def event97475 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9599⟩⟩, .relation 97474 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩)

def event97476 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9599⟩⟩, .operator (⟨97467, 0⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact97477RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩]

theorem exact97477RawTermsValid :
    exact97477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97477 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9599⟩⟩) exact97477RawTerms .large 97470 (.finite 95420416) (some (97472))

def event97478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11744⟩⟩) 0 ⟨9599⟩ 97477

def event97479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11744⟩⟩) 1 ⟨11743⟩ 97447

def event97480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11744⟩⟩) (.sum [.predecessor 0 97478 .coefficient, .predecessor 1 97479 .coefficient])

def event97481 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11744⟩⟩, .operator (⟨97477, 1⟩, ⟨97447, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def event97482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11744⟩⟩) (.sum [.result 97477 .summary, .result 97447 .summary])

def exact97483RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97483RawTermsValid :
    exact97483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11744⟩⟩) exact97483RawTerms .large 97480 (.finite 95445376) (some (97482))

def event97484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25130⟩⟩) 0 ⟨11744⟩ 97483

def event97485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25130⟩⟩) 1 ⟨25129⟩ 97419

def event97486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25130⟩⟩) (.product (.predecessor 0 97484 .coefficient) (.predecessor 1 97485 .coefficient) (⟨false, false, none, none, none⟩))

def event97487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25130⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩) [⟨.result 97419 .coefficient, false, none⟩])

def event97488 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25130⟩⟩) (.product (.result 97483 .summary) (.transfer 97487) (⟨false, false, none, none, none⟩))

def event97489 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25130⟩⟩, .operator (⟨97483, 1⟩, ⟨97419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (-1)⟩)

def event97490 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25130⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25129⟩⟩) ⟨23074⟩ 97416)

def event97491 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25130⟩⟩, .relation 97490 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (-1)⟩)

def event97492 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25130⟩⟩, .operator (⟨97483, 0⟩, ⟨97419, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩)

def exact97493RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], [⟨.program ⟨214⟩, ⟨23074⟩⟩]⟩, (-1)⟩]

theorem exact97493RawTermsValid :
    exact97493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97493 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25130⟩⟩) exact97493RawTerms .large 97486 (.finite 350286057046016) (some (97488))

def event97494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19733⟩⟩) 0 ⟨11739⟩ 4738

def event97495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19733⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact97496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩]

theorem exact97496RawTermsValid :
    exact97496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19733⟩⟩) exact97496RawTerms (.finite 136065468) 97495 .exactZero (none)

def event97497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19735⟩⟩) 0 ⟨19733⟩ 97496

def event97498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19735⟩⟩) 1 ⟨2348⟩ 4

def event97499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19735⟩⟩) (.scale (.predecessor 0 97497 .coefficient) (.value (.predecessor 1 97498 .coefficient)))

def exact97500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩]

theorem exact97500RawTermsValid :
    exact97500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19735⟩⟩) exact97500RawTerms (.finite 136065468) 97499 .exactZero (none)

def event97501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19736⟩⟩) 0 ⟨5509⟩ 94462

def event97502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19736⟩⟩) 1 ⟨19735⟩ 97500

def event97503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19736⟩⟩) (.product (.predecessor 0 97501 .coefficient) (.predecessor 1 97502 .coefficient) (⟨false, false, none, none, none⟩))

def event97504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19736⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩) [⟨.result 97496 .coefficient, false, none⟩])

def event97505 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19736⟩⟩) (.product (.result 94462 .summary) (.transfer 97504) (⟨false, false, none, none, none⟩))

def event97506 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19736⟩⟩, .operator (⟨94462, 0⟩, ⟨97500, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩)

def event97507 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19734⟩⟩)

def event97508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97509 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97511

def event97513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97509

def event97514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97512 .coefficient) (.value (.predecessor 1 97513 .coefficient)))

def event97515 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11737⟩⟩) 0 ⟨5503⟩ 97515

def event97517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11737⟩⟩) (.authority (.programFamilyFact))

def exact97518RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩, (1)⟩]

theorem exact97518RawTermsValid :
    exact97518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97518 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11737⟩⟩) exact97518RawTerms (.finite 30) 97517 .exactZero (none)

def event97519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9595⟩⟩) 0 ⟨5503⟩ 97515

def event97520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9595⟩⟩) (.authority (.programFamilyFact))

def exact97521RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩, (1)⟩]

theorem exact97521RawTermsValid :
    exact97521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9595⟩⟩) exact97521RawTerms (.finite 30) 97520 .exactZero (none)

def event97522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 0 ⟨9595⟩ 97521

def event97523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11738⟩⟩) 1 ⟨11737⟩ 97518

def event97524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.product (.predecessor 0 97522 .coefficient) (.predecessor 1 97523 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11738⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩, ⟨.program ⟨214⟩, ⟨11737⟩⟩], []⟩) [⟨.result 97521 .coefficient, true, some 1⟩, ⟨.result 97518 .coefficient, true, some 1⟩])

def event97526 : Event := .survivorFold (1) 97525

def exact97527RawTerms : List Term := []

theorem exact97527RawTermsValid :
    exact97527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97527 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11738⟩⟩) exact97527RawTerms (.finite 900) 97524 (.finite 900) (some (97525))

def event97528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11739⟩⟩) 0 ⟨11738⟩ 97527

def event97529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.identity (.predecessor 0 97528 .coefficient))

def event97530 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11739⟩⟩) (.finite 900)

def event97531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19733⟩⟩) 0 ⟨11739⟩ 97530

def event97532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19733⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact97533RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19733⟩⟩]⟩, (1)⟩]

theorem exact97533RawTermsValid :
    exact97533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19733⟩⟩) exact97533RawTerms (.finite 136065468) 97532 .exactZero (none)

def event97534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact97535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact97535RawTermsValid :
    exact97535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact97535RawTerms .large 97534 .exactZero (none)

def eventLeaf6080 : Array AnnotatedEvent := #[
  { event := event97280
    frameStart := 97252 },
  { event := event97281
    frameStart := 97252 },
  { event := event97282
    frameStart := 97252 },
  { event := event97283
    frameStart := 97252 },
  { event := event97284
    frameStart := 97252 },
  { event := event97285
    frameStart := 97252 },
  { event := event97286
    frameStart := 97252 },
  { event := event97287
    frameStart := 97252 },
  { event := event97288
    frameStart := 97252 },
  { event := event97289
    frameStart := 97252 },
  { event := event97290
    frameStart := 97252 },
  { event := event97291
    frameStart := 97252 },
  { event := event97292
    frameStart := 97252 },
  { event := event97293
    frameStart := 97252 },
  { event := event97294
    frameStart := 97294 },
  { event := event97295
    frameStart := 97294 }
]

def eventLeaf6081 : Array AnnotatedEvent := #[
  { event := event97296
    frameStart := 97294 },
  { event := event97297
    frameStart := 97294 },
  { event := event97298
    frameStart := 97294 },
  { event := event97299
    frameStart := 97294 },
  { event := event97300
    frameStart := 97294 },
  { event := event97301
    frameStart := 97294 },
  { event := event97302
    frameStart := 97294 },
  { event := event97303
    frameStart := 97294 },
  { event := event97304
    frameStart := 97294 },
  { event := event97305
    frameStart := 97294 },
  { event := event97306
    frameStart := 97294 },
  { event := event97307
    frameStart := 97294 },
  { event := event97308
    frameStart := 97294 },
  { event := event97309
    frameStart := 97294 },
  { event := event97310
    frameStart := 97294 },
  { event := event97311
    frameStart := 97294 }
]

def eventLeaf6082 : Array AnnotatedEvent := #[
  { event := event97312
    frameStart := 97294 },
  { event := event97313
    frameStart := 97294 },
  { event := event97314
    frameStart := 97294 },
  { event := event97315
    frameStart := 97294 },
  { event := event97316
    frameStart := 97294 },
  { event := event97317
    frameStart := 97294 },
  { event := event97318
    frameStart := 97294 },
  { event := event97319
    frameStart := 97294 },
  { event := event97320
    frameStart := 97294 },
  { event := event97321
    frameStart := 97294 },
  { event := event97322
    frameStart := 97294 },
  { event := event97323
    frameStart := 97294 },
  { event := event97324
    frameStart := 97294 },
  { event := event97325
    frameStart := 97294 },
  { event := event97326
    frameStart := 97294 },
  { event := event97327
    frameStart := 97294 }
]

def eventLeaf6083 : Array AnnotatedEvent := #[
  { event := event97328
    frameStart := 97294 },
  { event := event97329
    frameStart := 97294 },
  { event := event97330
    frameStart := 97294 },
  { event := event97331
    frameStart := 97294 },
  { event := event97332
    frameStart := 97294 },
  { event := event97333
    frameStart := 97294 },
  { event := event97334
    frameStart := 97294 },
  { event := event97335
    frameStart := 97294 },
  { event := event97336
    frameStart := 97294 },
  { event := event97337
    frameStart := 97294 },
  { event := event97338
    frameStart := 97294 },
  { event := event97339
    frameStart := 97294 },
  { event := event97340
    frameStart := 97294 },
  { event := event97341
    frameStart := 97294 },
  { event := event97342
    frameStart := 97294 },
  { event := event97343
    frameStart := 97294 }
]

def eventLeaf6084 : Array AnnotatedEvent := #[
  { event := event97344
    frameStart := 97294 },
  { event := event97345
    frameStart := 97294 },
  { event := event97346
    frameStart := 97294 },
  { event := event97347
    frameStart := 97294 },
  { event := event97348
    frameStart := 97294 },
  { event := event97349
    frameStart := 97294 },
  { event := event97350
    frameStart := 97294 },
  { event := event97351
    frameStart := 97294 },
  { event := event97352
    frameStart := 97294 },
  { event := event97353
    frameStart := 97294 },
  { event := event97354
    frameStart := 97294 },
  { event := event97355
    frameStart := 97294 },
  { event := event97356
    frameStart := 97294 },
  { event := event97357
    frameStart := 97294 },
  { event := event97358
    frameStart := 97294 },
  { event := event97359
    frameStart := 97294 }
]

def eventLeaf6085 : Array AnnotatedEvent := #[
  { event := event97360
    frameStart := 97294 },
  { event := event97361
    frameStart := 97294 },
  { event := event97362
    frameStart := 97294 },
  { event := event97363
    frameStart := 97294 },
  { event := event97364
    frameStart := 97294 },
  { event := event97365
    frameStart := 97294 },
  { event := event97366
    frameStart := 97294 },
  { event := event97367
    frameStart := 97294 },
  { event := event97368
    frameStart := 97294 },
  { event := event97369
    frameStart := 97294 },
  { event := event97370
    frameStart := 97294 },
  { event := event97371
    frameStart := 97294 },
  { event := event97372
    frameStart := 97294 },
  { event := event97373
    frameStart := 97294 },
  { event := event97374
    frameStart := 97294 },
  { event := event97375
    frameStart := 97294 }
]

def eventLeaf6086 : Array AnnotatedEvent := #[
  { event := event97376
    frameStart := 97294 },
  { event := event97377
    frameStart := 97294 },
  { event := event97378
    frameStart := 97294 },
  { event := event97379
    frameStart := 97294 },
  { event := event97380
    frameStart := 97294 },
  { event := event97381
    frameStart := 97294 },
  { event := event97382
    frameStart := 97294 },
  { event := event97383
    frameStart := 97294 },
  { event := event97384
    frameStart := 97294 },
  { event := event97385
    frameStart := 97294 },
  { event := event97386
    frameStart := 0 },
  { event := event97387
    frameStart := 0 },
  { event := event97388
    frameStart := 0 },
  { event := event97389
    frameStart := 0 },
  { event := event97390
    frameStart := 0 },
  { event := event97391
    frameStart := 0 }
]

def eventLeaf6087 : Array AnnotatedEvent := #[
  { event := event97392
    frameStart := 0 },
  { event := event97393
    frameStart := 0 },
  { event := event97394
    frameStart := 0 },
  { event := event97395
    frameStart := 0 },
  { event := event97396
    frameStart := 0 },
  { event := event97397
    frameStart := 0 },
  { event := event97398
    frameStart := 0 },
  { event := event97399
    frameStart := 0 },
  { event := event97400
    frameStart := 0 },
  { event := event97401
    frameStart := 0 },
  { event := event97402
    frameStart := 0 },
  { event := event97403
    frameStart := 0 },
  { event := event97404
    frameStart := 0 },
  { event := event97405
    frameStart := 0 },
  { event := event97406
    frameStart := 0 },
  { event := event97407
    frameStart := 0 }
]

def eventLeaf6088 : Array AnnotatedEvent := #[
  { event := event97408
    frameStart := 0 },
  { event := event97409
    frameStart := 0 },
  { event := event97410
    frameStart := 0 },
  { event := event97411
    frameStart := 0 },
  { event := event97412
    frameStart := 0 },
  { event := event97413
    frameStart := 0 },
  { event := event97414
    frameStart := 0 },
  { event := event97415
    frameStart := 0 },
  { event := event97416
    frameStart := 0 },
  { event := event97417
    frameStart := 0 },
  { event := event97418
    frameStart := 0 },
  { event := event97419
    frameStart := 0 },
  { event := event97420
    frameStart := 0 },
  { event := event97421
    frameStart := 0 },
  { event := event97422
    frameStart := 0 },
  { event := event97423
    frameStart := 0 }
]

def eventLeaf6089 : Array AnnotatedEvent := #[
  { event := event97424
    frameStart := 0 },
  { event := event97425
    frameStart := 0 },
  { event := event97426
    frameStart := 0 },
  { event := event97427
    frameStart := 0 },
  { event := event97428
    frameStart := 0 },
  { event := event97429
    frameStart := 0 },
  { event := event97430
    frameStart := 0 },
  { event := event97431
    frameStart := 0 },
  { event := event97432
    frameStart := 0 },
  { event := event97433
    frameStart := 0 },
  { event := event97434
    frameStart := 0 },
  { event := event97435
    frameStart := 0 },
  { event := event97436
    frameStart := 0 },
  { event := event97437
    frameStart := 0 },
  { event := event97438
    frameStart := 0 },
  { event := event97439
    frameStart := 0 }
]

def eventLeaf6090 : Array AnnotatedEvent := #[
  { event := event97440
    frameStart := 0 },
  { event := event97441
    frameStart := 0 },
  { event := event97442
    frameStart := 0 },
  { event := event97443
    frameStart := 0 },
  { event := event97444
    frameStart := 0 },
  { event := event97445
    frameStart := 0 },
  { event := event97446
    frameStart := 0 },
  { event := event97447
    frameStart := 0 },
  { event := event97448
    frameStart := 0 },
  { event := event97449
    frameStart := 0 },
  { event := event97450
    frameStart := 0 },
  { event := event97451
    frameStart := 0 },
  { event := event97452
    frameStart := 0 },
  { event := event97453
    frameStart := 0 },
  { event := event97454
    frameStart := 0 },
  { event := event97455
    frameStart := 0 }
]

def eventLeaf6091 : Array AnnotatedEvent := #[
  { event := event97456
    frameStart := 0 },
  { event := event97457
    frameStart := 0 },
  { event := event97458
    frameStart := 0 },
  { event := event97459
    frameStart := 0 },
  { event := event97460
    frameStart := 0 },
  { event := event97461
    frameStart := 0 },
  { event := event97462
    frameStart := 0 },
  { event := event97463
    frameStart := 0 },
  { event := event97464
    frameStart := 0 },
  { event := event97465
    frameStart := 0 },
  { event := event97466
    frameStart := 0 },
  { event := event97467
    frameStart := 0 },
  { event := event97468
    frameStart := 0 },
  { event := event97469
    frameStart := 0 },
  { event := event97470
    frameStart := 0 },
  { event := event97471
    frameStart := 0 }
]

def eventLeaf6092 : Array AnnotatedEvent := #[
  { event := event97472
    frameStart := 0 },
  { event := event97473
    frameStart := 0 },
  { event := event97474
    frameStart := 0 },
  { event := event97475
    frameStart := 0 },
  { event := event97476
    frameStart := 0 },
  { event := event97477
    frameStart := 0 },
  { event := event97478
    frameStart := 0 },
  { event := event97479
    frameStart := 0 },
  { event := event97480
    frameStart := 0 },
  { event := event97481
    frameStart := 0 },
  { event := event97482
    frameStart := 0 },
  { event := event97483
    frameStart := 0 },
  { event := event97484
    frameStart := 0 },
  { event := event97485
    frameStart := 0 },
  { event := event97486
    frameStart := 0 },
  { event := event97487
    frameStart := 0 }
]

def eventLeaf6093 : Array AnnotatedEvent := #[
  { event := event97488
    frameStart := 0 },
  { event := event97489
    frameStart := 0 },
  { event := event97490
    frameStart := 0 },
  { event := event97491
    frameStart := 0 },
  { event := event97492
    frameStart := 0 },
  { event := event97493
    frameStart := 0 },
  { event := event97494
    frameStart := 0 },
  { event := event97495
    frameStart := 0 },
  { event := event97496
    frameStart := 0 },
  { event := event97497
    frameStart := 0 },
  { event := event97498
    frameStart := 0 },
  { event := event97499
    frameStart := 0 },
  { event := event97500
    frameStart := 0 },
  { event := event97501
    frameStart := 0 },
  { event := event97502
    frameStart := 0 },
  { event := event97503
    frameStart := 0 }
]

def eventLeaf6094 : Array AnnotatedEvent := #[
  { event := event97504
    frameStart := 0 },
  { event := event97505
    frameStart := 0 },
  { event := event97506
    frameStart := 0 },
  { event := event97507
    frameStart := 97507 },
  { event := event97508
    frameStart := 97507 },
  { event := event97509
    frameStart := 97507 },
  { event := event97510
    frameStart := 97507 },
  { event := event97511
    frameStart := 97507 },
  { event := event97512
    frameStart := 97507 },
  { event := event97513
    frameStart := 97507 },
  { event := event97514
    frameStart := 97507 },
  { event := event97515
    frameStart := 97507 },
  { event := event97516
    frameStart := 97507 },
  { event := event97517
    frameStart := 97507 },
  { event := event97518
    frameStart := 97507 },
  { event := event97519
    frameStart := 97507 }
]

def eventLeaf6095 : Array AnnotatedEvent := #[
  { event := event97520
    frameStart := 97507 },
  { event := event97521
    frameStart := 97507 },
  { event := event97522
    frameStart := 97507 },
  { event := event97523
    frameStart := 97507 },
  { event := event97524
    frameStart := 97507 },
  { event := event97525
    frameStart := 97507 },
  { event := event97526
    frameStart := 97507 },
  { event := event97527
    frameStart := 97507 },
  { event := event97528
    frameStart := 97507 },
  { event := event97529
    frameStart := 97507 },
  { event := event97530
    frameStart := 97507 },
  { event := event97531
    frameStart := 97507 },
  { event := event97532
    frameStart := 97507 },
  { event := event97533
    frameStart := 97507 },
  { event := event97534
    frameStart := 97507 },
  { event := event97535
    frameStart := 97507 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events380
