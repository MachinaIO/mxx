import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events341

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event87296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.product (.predecessor 0 87294 .coefficient) (.predecessor 1 87295 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87297 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10978⟩⟩, .operator (⟨87293, 0⟩, ⟨87290, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩)

def exact87298RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact87298RawTermsValid :
    exact87298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87298 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10978⟩⟩) exact87298RawTerms (.finite 16) 87296 .exactZero (none)

def event87299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10979⟩⟩) 0 ⟨10978⟩ 87298

def event87300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.identity (.predecessor 0 87299 .coefficient))

def event87301 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.finite 16)

def event87302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23037⟩⟩) 0 ⟨10979⟩ 87301

def event87303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23037⟩⟩) (.authority (.programFamilyFact))

def event87304 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23037⟩⟩) (.finite 3720)

def event87305 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event87306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23038⟩⟩) 0 ⟨6689⟩ 87305

def event87307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23038⟩⟩) 1 ⟨23037⟩ 87304

def event87308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23038⟩⟩) (.authority (.operator))

def exact87309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩]

theorem exact87309RawTermsValid :
    exact87309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87309 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23038⟩⟩) exact87309RawTerms .large 87308 .exactZero (none)

def event87310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25065⟩⟩) 0 ⟨23038⟩ 87309

def event87311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25065⟩⟩) (.authority (.operator))

def exact87312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (1)⟩]

theorem exact87312RawTermsValid :
    exact87312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25065⟩⟩) exact87312RawTerms (.finite 8192) 87311 .exactZero (none)

def event87313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event87314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event87315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11073⟩⟩) 0 ⟨10979⟩ 87301

def event87316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11073⟩⟩) 1 ⟨110⟩ 87314

def event87317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11073⟩⟩) (.sum [.predecessor 0 87315 .coefficient, .predecessor 1 87316 .coefficient])

def event87318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11073⟩⟩) (.finite 16)

def event87319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11074⟩⟩) 0 ⟨11073⟩ 87318

def event87320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11074⟩⟩) (.identity (.predecessor 0 87319 .coefficient))

def exact87321RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact87321RawTermsValid :
    exact87321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11074⟩⟩) exact87321RawTerms (.finite 16) 87320 .exactZero (none)

def event87322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact87323RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87323RawTermsValid :
    exact87323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87323 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact87323RawTerms .large 87322 .exactZero (none)

def event87324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11075⟩⟩) 0 ⟨6544⟩ 87323

def event87325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11075⟩⟩) 1 ⟨11074⟩ 87321

def event87326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11075⟩⟩) (.product (.predecessor 0 87324 .coefficient) (.predecessor 1 87325 .coefficient) (⟨false, false, none, none, none⟩))

def event87327 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11075⟩⟩, .operator (⟨87323, 0⟩, ⟨87321, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact87328RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87328RawTermsValid :
    exact87328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11075⟩⟩) exact87328RawTerms .large 87326 .exactZero (none)

def event87329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 87305

def event87330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact87331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact87331RawTermsValid :
    exact87331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact87331RawTerms .large 87330 .exactZero (none)

def event87332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6774⟩⟩) 0 ⟨6757⟩ 87331

def event87333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6774⟩⟩) (.identity (.predecessor 0 87332 .coefficient))

def exact87334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩, (1)⟩]

theorem exact87334RawTermsValid :
    exact87334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87334 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6774⟩⟩) exact87334RawTerms .large 87333 .exactZero (none)

def event87335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7837⟩⟩) 0 ⟨6774⟩ 87334

def event87336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7837⟩⟩) (.authority (.operator))

def exact87337RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact87337RawTermsValid :
    exact87337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87337 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7837⟩⟩) exact87337RawTerms (.finite 8192) 87336 .exactZero (none)

def event87338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7838⟩⟩) 0 ⟨7837⟩ 87337

def event87339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7838⟩⟩) 1 ⟨2348⟩ 87271

def event87340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7838⟩⟩) (.scale (.predecessor 0 87338 .coefficient) (.value (.predecessor 1 87339 .coefficient)))

def exact87341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact87341RawTermsValid :
    exact87341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7838⟩⟩) exact87341RawTerms (.finite 8192) 87340 .exactZero (none)

def event87342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6791⟩⟩) 0 ⟨6757⟩ 87331

def event87343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6791⟩⟩) (.identity (.predecessor 0 87342 .coefficient))

def exact87344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩]⟩, (1)⟩]

theorem exact87344RawTermsValid :
    exact87344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6791⟩⟩) exact87344RawTerms .large 87343 .exactZero (none)

def event87345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7839⟩⟩) 0 ⟨6791⟩ 87344

def event87346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7839⟩⟩) 1 ⟨7838⟩ 87341

def event87347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7839⟩⟩) (.product (.predecessor 0 87345 .coefficient) (.predecessor 1 87346 .coefficient) (⟨false, false, none, none, none⟩))

def event87348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7839⟩⟩, .operator (⟨87344, 0⟩, ⟨87341, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩)

def exact87349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact87349RawTermsValid :
    exact87349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87349 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7839⟩⟩) exact87349RawTerms .large 87347 .exactZero (none)

def event87350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11076⟩⟩) 0 ⟨7839⟩ 87349

def event87351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11076⟩⟩) 1 ⟨11075⟩ 87328

def event87352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11076⟩⟩) (.sum [.predecessor 0 87350 .coefficient, .predecessor 1 87351 .coefficient])

def exact87353RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87353RawTermsValid :
    exact87353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87353 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11076⟩⟩) exact87353RawTerms .large 87352 .exactZero (none)

def event87354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25068⟩⟩) 0 ⟨11076⟩ 87353

def event87355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25068⟩⟩) 1 ⟨25065⟩ 87312

def event87356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25068⟩⟩) (.product (.predecessor 0 87354 .coefficient) (.predecessor 1 87355 .coefficient) (⟨false, false, none, none, none⟩))

def event87357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25068⟩⟩, .operator (⟨87353, 0⟩, ⟨87312, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (1)⟩)

def event87358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25068⟩⟩, .operator (⟨87353, 1⟩, ⟨87312, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩)

def event87359 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25068⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25065⟩⟩) ⟨23038⟩ 87309)

def event87360 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25068⟩⟩, .relation 87359 0, ⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (-1)⟩)

def exact87361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (-1)⟩]

theorem exact87361RawTermsValid :
    exact87361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25068⟩⟩) exact87361RawTerms .large 87356 .exactZero (none)

def event87362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15114⟩⟩) 0 ⟨10979⟩ 87301

def event87363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15114⟩⟩) (.authority (.programFamilyFact))

def exact87364RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], []⟩, (1)⟩]

theorem exact87364RawTermsValid :
    exact87364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15114⟩⟩) exact87364RawTerms (.finite 4) 87363 .exactZero (none)

def event87365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15116⟩⟩) 0 ⟨6544⟩ 87323

def event87366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15116⟩⟩) 1 ⟨15114⟩ 87364

def event87367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15116⟩⟩) (.product (.predecessor 0 87365 .coefficient) (.predecessor 1 87366 .coefficient) (⟨false, true, none, none, some 1⟩))

def event87368 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15116⟩⟩, .operator (⟨87323, 0⟩, ⟨87364, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact87369RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87369RawTermsValid :
    exact87369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15116⟩⟩) exact87369RawTerms .large 87367 .exactZero (none)

def event87370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6692⟩⟩) 0 ⟨6689⟩ 87305

def event87371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6692⟩⟩) (.authority (.operator))

def exact87372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩]

theorem exact87372RawTermsValid :
    exact87372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6692⟩⟩) exact87372RawTerms .large 87371 .exactZero (none)

def event87373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15117⟩⟩) 0 ⟨6692⟩ 87372

def event87374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15117⟩⟩) 1 ⟨15116⟩ 87369

def event87375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15117⟩⟩) (.sum [.predecessor 0 87373 .coefficient, .predecessor 1 87374 .coefficient])

def exact87376RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87376RawTermsValid :
    exact87376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15117⟩⟩) exact87376RawTerms .large 87375 .exactZero (none)

def event87377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25069⟩⟩) 0 ⟨15117⟩ 87376

def event87378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25069⟩⟩) 1 ⟨25068⟩ 87361

def event87379 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25069⟩⟩) (.sum [.predecessor 0 87377 .coefficient, .predecessor 1 87378 .coefficient])

def exact87380RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87380RawTermsValid :
    exact87380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87380 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25069⟩⟩) exact87380RawTerms .large 87379 .exactZero (none)

def event87381 : Event := .preFoldPolynomial 87380 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact87382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event87382 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25069⟩⟩) 87381 exact87382RawTerms .large 87379 .exactZero (none)

def event87383 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10979⟩⟩) ⟨⟨105⟩, ⟨9⟩, ⟨109⟩⟩ ⟨87219, 87383⟩

def event87384 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19171⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19168⟩⟩]⟩) (1) 0 2 (.universal 87383 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19168⟩⟩]⟩) (none) 87382)

def event87385 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19171⟩⟩, .relation 87384 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩)

def event87386 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19171⟩⟩, .relation 87384 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩)

def event87387 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19171⟩⟩, .relation 87384 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩)

def event87388 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19171⟩⟩, .relation 87384 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact87389RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87389RawTermsValid :
    exact87389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87389 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19171⟩⟩) exact87389RawTerms .large 87215 (.finite 1811303510016) (some (87217))

def event87390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25067⟩⟩) 0 ⟨19171⟩ 87389

def event87391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25067⟩⟩) 1 ⟨25066⟩ 87205

def event87392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25067⟩⟩) (.sum [.predecessor 0 87390 .coefficient, .predecessor 1 87391 .coefficient])

def event87393 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25067⟩⟩, .operator (⟨87389, 2⟩, ⟨87205, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨23038⟩⟩]⟩, (-1)⟩)

def event87394 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25067⟩⟩, .operator (⟨87389, 1⟩, ⟨87205, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩, (1)⟩)

def event87395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25067⟩⟩) (.sum [.result 87389 .summary, .result 87205 .summary])

def exact87396RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87396RawTermsValid :
    exact87396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87396 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25067⟩⟩) exact87396RawTerms .large 87392 (.finite 352017970769920) (some (87395))

def event87397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26783⟩⟩) 0 ⟨25067⟩ 87396

def event87398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26783⟩⟩) 1 ⟨26781⟩ 87121

def event87399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26783⟩⟩) (.product (.predecessor 0 87397 .coefficient) (.predecessor 1 87398 .coefficient) (⟨false, false, none, none, none⟩))

def event87400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26783⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩) [⟨.result 87121 .coefficient, false, none⟩])

def event87401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26783⟩⟩) (.product (.result 87396 .summary) (.transfer 87400) (⟨false, false, none, none, none⟩))

def event87402 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26783⟩⟩, .operator (⟨87396, 0⟩, ⟨87121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩, (1)⟩)

def event87403 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26783⟩⟩, .operator (⟨87396, 1⟩, ⟨87121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩, (-1)⟩)

def event87404 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26783⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26781⟩⟩) ⟨23847⟩ 87118)

def event87405 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26783⟩⟩, .relation 87404 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨23847⟩⟩]⟩, (-1)⟩)

def exact87406RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨23847⟩⟩]⟩, (-1)⟩]

theorem exact87406RawTermsValid :
    exact87406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26783⟩⟩) exact87406RawTerms .large 87399 (.finite 1291911585013138718720) (some (87401))

def event87407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20680⟩⟩) 0 ⟨15115⟩ 4190

def event87408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20680⟩⟩) (.authority (.relationPreimageSource ⟨32⟩))

def exact87409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩]

theorem exact87409RawTermsValid :
    exact87409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20680⟩⟩) exact87409RawTerms (.finite 136065468) 87408 .exactZero (none)

def event87410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20682⟩⟩) 0 ⟨20680⟩ 87409

def event87411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20682⟩⟩) 1 ⟨2348⟩ 4

def event87412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20682⟩⟩) (.scale (.predecessor 0 87410 .coefficient) (.value (.predecessor 1 87411 .coefficient)))

def exact87413RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩]

theorem exact87413RawTermsValid :
    exact87413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20682⟩⟩) exact87413RawTerms (.finite 136065468) 87412 .exactZero (none)

def event87414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20683⟩⟩) 0 ⟨5541⟩ 80012

def event87415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20683⟩⟩) 1 ⟨20682⟩ 87413

def event87416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20683⟩⟩) (.product (.predecessor 0 87414 .coefficient) (.predecessor 1 87415 .coefficient) (⟨false, false, none, none, none⟩))

def event87417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20683⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩) [⟨.result 87409 .coefficient, false, none⟩])

def event87418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20683⟩⟩) (.product (.result 80012 .summary) (.transfer 87417) (⟨false, false, none, none, none⟩))

def event87419 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20683⟩⟩, .operator (⟨80012, 0⟩, ⟨87413, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩)

def event87420 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20681⟩⟩)

def event87421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event87422 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event87423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event87424 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event87425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event87426 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event87427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event87428 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event87429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 87428

def event87430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 87426

def event87431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 87429 .coefficient) (.value (.predecessor 1 87430 .coefficient)))

def event87432 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event87433 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 87432

def event87434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 87424

def event87435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 87433 .coefficient, .predecessor 1 87434 .coefficient])

def event87436 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event87437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 87436

def event87438 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 87422

def event87439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 87438 .coefficient))

def event87440 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event87441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10977⟩⟩) 0 ⟨5536⟩ 87440

def event87442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10977⟩⟩) (.authority (.programFamilyFact))

def exact87443RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact87443RawTermsValid :
    exact87443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10977⟩⟩) exact87443RawTerms (.finite 4) 87442 .exactZero (none)

def event87444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10842⟩⟩) 0 ⟨5536⟩ 87440

def event87445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10842⟩⟩) (.authority (.programFamilyFact))

def exact87446RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩], []⟩, (1)⟩]

theorem exact87446RawTermsValid :
    exact87446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10842⟩⟩) exact87446RawTerms (.finite 4) 87445 .exactZero (none)

def event87447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 0 ⟨10842⟩ 87446

def event87448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 1 ⟨10977⟩ 87443

def event87449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.product (.predecessor 0 87447 .coefficient) (.predecessor 1 87448 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩) [⟨.result 87446 .coefficient, true, some 1⟩, ⟨.result 87443 .coefficient, true, some 1⟩])

def event87451 : Event := .survivorFold (1) 87450

def exact87452RawTerms : List Term := []

theorem exact87452RawTermsValid :
    exact87452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87452 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10978⟩⟩) exact87452RawTerms (.finite 16) 87449 (.finite 16) (some (87450))

def event87453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10979⟩⟩) 0 ⟨10978⟩ 87452

def event87454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.identity (.predecessor 0 87453 .coefficient))

def event87455 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.finite 16)

def event87456 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15114⟩⟩) 0 ⟨10979⟩ 87455

def event87457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15114⟩⟩) (.authority (.programFamilyFact))

def exact87458RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], []⟩, (1)⟩]

theorem exact87458RawTermsValid :
    exact87458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87458 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15114⟩⟩) exact87458RawTerms (.finite 4) 87457 .exactZero (none)

def event87459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15115⟩⟩) 0 ⟨15114⟩ 87458

def event87460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.identity (.predecessor 0 87459 .coefficient))

def event87461 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.finite 4)

def event87462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20680⟩⟩) 0 ⟨15115⟩ 87461

def event87463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20680⟩⟩) (.authority (.relationPreimageSource ⟨32⟩))

def exact87464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩]

theorem exact87464RawTermsValid :
    exact87464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20680⟩⟩) exact87464RawTerms (.finite 136065468) 87463 .exactZero (none)

def event87465 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact87466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact87466RawTermsValid :
    exact87466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87466 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact87466RawTerms .large 87465 .exactZero (none)

def event87467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20681⟩⟩) 0 ⟨6⟩ 87466

def event87468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20681⟩⟩) 1 ⟨20680⟩ 87464

def event87469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20681⟩⟩) (.product (.predecessor 0 87467 .coefficient) (.predecessor 1 87468 .coefficient) (⟨false, false, none, none, none⟩))

def event87470 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20681⟩⟩, .operator (⟨87466, 0⟩, ⟨87464, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩)

def exact87471RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩]

theorem exact87471RawTermsValid :
    exact87471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20681⟩⟩) exact87471RawTerms .large 87469 .exactZero (none)

def event87472 : Event := .preFoldPolynomial 87471 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩] .exactZero none

def exact87473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩, (1)⟩]

def event87473 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20681⟩⟩) 87472 exact87473RawTerms .large 87469 .exactZero (none)

def event87474 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26786⟩⟩)

def event87475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event87476 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event87477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event87478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event87479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event87480 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event87481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event87482 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event87483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 87482

def event87484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 87480

def event87485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 87483 .coefficient) (.value (.predecessor 1 87484 .coefficient)))

def event87486 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event87487 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 87486

def event87488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 87478

def event87489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 87487 .coefficient, .predecessor 1 87488 .coefficient])

def event87490 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event87491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 87490

def event87492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 87476

def event87493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 87492 .coefficient))

def event87494 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event87495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10977⟩⟩) 0 ⟨5536⟩ 87494

def event87496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10977⟩⟩) (.authority (.programFamilyFact))

def exact87497RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact87497RawTermsValid :
    exact87497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10977⟩⟩) exact87497RawTerms (.finite 4) 87496 .exactZero (none)

def event87498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10842⟩⟩) 0 ⟨5536⟩ 87494

def event87499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10842⟩⟩) (.authority (.programFamilyFact))

def exact87500RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩], []⟩, (1)⟩]

theorem exact87500RawTermsValid :
    exact87500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10842⟩⟩) exact87500RawTerms (.finite 4) 87499 .exactZero (none)

def event87501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 0 ⟨10842⟩ 87500

def event87502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 1 ⟨10977⟩ 87497

def event87503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.product (.predecessor 0 87501 .coefficient) (.predecessor 1 87502 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87504 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10978⟩⟩, .operator (⟨87500, 0⟩, ⟨87497, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩)

def exact87505RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact87505RawTermsValid :
    exact87505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10978⟩⟩) exact87505RawTerms (.finite 16) 87503 .exactZero (none)

def event87506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10979⟩⟩) 0 ⟨10978⟩ 87505

def event87507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.identity (.predecessor 0 87506 .coefficient))

def event87508 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.finite 16)

def event87509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15114⟩⟩) 0 ⟨10979⟩ 87508

def event87510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15114⟩⟩) (.authority (.programFamilyFact))

def exact87511RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], []⟩, (1)⟩]

theorem exact87511RawTermsValid :
    exact87511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87511 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15114⟩⟩) exact87511RawTerms (.finite 4) 87510 .exactZero (none)

def event87512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15115⟩⟩) 0 ⟨15114⟩ 87511

def event87513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.identity (.predecessor 0 87512 .coefficient))

def event87514 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.finite 4)

def event87515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23845⟩⟩) 0 ⟨15115⟩ 87514

def event87516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23845⟩⟩) (.authority (.programFamilyFact))

def event87517 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23845⟩⟩) (.finite 3720)

def event87518 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event87519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23847⟩⟩) 0 ⟨6689⟩ 87518

def event87520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23847⟩⟩) 1 ⟨23845⟩ 87517

def event87521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23847⟩⟩) (.authority (.operator))

def exact87522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23847⟩⟩]⟩, (1)⟩]

theorem exact87522RawTermsValid :
    exact87522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23847⟩⟩) exact87522RawTerms .large 87521 .exactZero (none)

def event87523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26781⟩⟩) 0 ⟨23847⟩ 87522

def event87524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26781⟩⟩) (.authority (.operator))

def exact87525RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩, (1)⟩]

theorem exact87525RawTermsValid :
    exact87525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26781⟩⟩) exact87525RawTerms (.finite 8192) 87524 .exactZero (none)

def event87526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event87527 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event87528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15154⟩⟩) 0 ⟨15115⟩ 87514

def event87529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15154⟩⟩) 1 ⟨110⟩ 87527

def event87530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15154⟩⟩) (.sum [.predecessor 0 87528 .coefficient, .predecessor 1 87529 .coefficient])

def event87531 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15154⟩⟩) (.finite 4)

def event87532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15155⟩⟩) 0 ⟨15154⟩ 87531

def event87533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15155⟩⟩) (.identity (.predecessor 0 87532 .coefficient))

def exact87534RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], []⟩, (1)⟩]

theorem exact87534RawTermsValid :
    exact87534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87534 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15155⟩⟩) exact87534RawTerms (.finite 4) 87533 .exactZero (none)

def event87535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact87536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87536RawTermsValid :
    exact87536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87536 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact87536RawTerms .large 87535 .exactZero (none)

def event87537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15156⟩⟩) 0 ⟨6544⟩ 87536

def event87538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15156⟩⟩) 1 ⟨15155⟩ 87534

def event87539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15156⟩⟩) (.product (.predecessor 0 87537 .coefficient) (.predecessor 1 87538 .coefficient) (⟨false, false, none, none, none⟩))

def event87540 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15156⟩⟩, .operator (⟨87536, 0⟩, ⟨87534, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact87541RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact87541RawTermsValid :
    exact87541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87541 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15156⟩⟩) exact87541RawTerms .large 87539 .exactZero (none)

def event87542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6692⟩⟩) 0 ⟨6689⟩ 87518

def event87543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6692⟩⟩) (.authority (.operator))

def exact87544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩]

theorem exact87544RawTermsValid :
    exact87544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6692⟩⟩) exact87544RawTerms .large 87543 .exactZero (none)

def event87545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15157⟩⟩) 0 ⟨6692⟩ 87544

def event87546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15157⟩⟩) 1 ⟨15156⟩ 87541

def event87547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15157⟩⟩) (.sum [.predecessor 0 87545 .coefficient, .predecessor 1 87546 .coefficient])

def exact87548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact87548RawTermsValid :
    exact87548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87548 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15157⟩⟩) exact87548RawTerms .large 87547 .exactZero (none)

def event87549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26782⟩⟩) 0 ⟨15157⟩ 87548

def event87550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26782⟩⟩) 1 ⟨26781⟩ 87525

def event87551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26782⟩⟩) (.product (.predecessor 0 87549 .coefficient) (.predecessor 1 87550 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf5456 : Array AnnotatedEvent := #[
  { event := event87296
    frameStart := 87267 },
  { event := event87297
    frameStart := 87267 },
  { event := event87298
    frameStart := 87267 },
  { event := event87299
    frameStart := 87267 },
  { event := event87300
    frameStart := 87267 },
  { event := event87301
    frameStart := 87267 },
  { event := event87302
    frameStart := 87267 },
  { event := event87303
    frameStart := 87267 },
  { event := event87304
    frameStart := 87267 },
  { event := event87305
    frameStart := 87267 },
  { event := event87306
    frameStart := 87267 },
  { event := event87307
    frameStart := 87267 },
  { event := event87308
    frameStart := 87267 },
  { event := event87309
    frameStart := 87267 },
  { event := event87310
    frameStart := 87267 },
  { event := event87311
    frameStart := 87267 }
]

def eventLeaf5457 : Array AnnotatedEvent := #[
  { event := event87312
    frameStart := 87267 },
  { event := event87313
    frameStart := 87267 },
  { event := event87314
    frameStart := 87267 },
  { event := event87315
    frameStart := 87267 },
  { event := event87316
    frameStart := 87267 },
  { event := event87317
    frameStart := 87267 },
  { event := event87318
    frameStart := 87267 },
  { event := event87319
    frameStart := 87267 },
  { event := event87320
    frameStart := 87267 },
  { event := event87321
    frameStart := 87267 },
  { event := event87322
    frameStart := 87267 },
  { event := event87323
    frameStart := 87267 },
  { event := event87324
    frameStart := 87267 },
  { event := event87325
    frameStart := 87267 },
  { event := event87326
    frameStart := 87267 },
  { event := event87327
    frameStart := 87267 }
]

def eventLeaf5458 : Array AnnotatedEvent := #[
  { event := event87328
    frameStart := 87267 },
  { event := event87329
    frameStart := 87267 },
  { event := event87330
    frameStart := 87267 },
  { event := event87331
    frameStart := 87267 },
  { event := event87332
    frameStart := 87267 },
  { event := event87333
    frameStart := 87267 },
  { event := event87334
    frameStart := 87267 },
  { event := event87335
    frameStart := 87267 },
  { event := event87336
    frameStart := 87267 },
  { event := event87337
    frameStart := 87267 },
  { event := event87338
    frameStart := 87267 },
  { event := event87339
    frameStart := 87267 },
  { event := event87340
    frameStart := 87267 },
  { event := event87341
    frameStart := 87267 },
  { event := event87342
    frameStart := 87267 },
  { event := event87343
    frameStart := 87267 }
]

def eventLeaf5459 : Array AnnotatedEvent := #[
  { event := event87344
    frameStart := 87267 },
  { event := event87345
    frameStart := 87267 },
  { event := event87346
    frameStart := 87267 },
  { event := event87347
    frameStart := 87267 },
  { event := event87348
    frameStart := 87267 },
  { event := event87349
    frameStart := 87267 },
  { event := event87350
    frameStart := 87267 },
  { event := event87351
    frameStart := 87267 },
  { event := event87352
    frameStart := 87267 },
  { event := event87353
    frameStart := 87267 },
  { event := event87354
    frameStart := 87267 },
  { event := event87355
    frameStart := 87267 },
  { event := event87356
    frameStart := 87267 },
  { event := event87357
    frameStart := 87267 },
  { event := event87358
    frameStart := 87267 },
  { event := event87359
    frameStart := 87267 }
]

def eventLeaf5460 : Array AnnotatedEvent := #[
  { event := event87360
    frameStart := 87267 },
  { event := event87361
    frameStart := 87267 },
  { event := event87362
    frameStart := 87267 },
  { event := event87363
    frameStart := 87267 },
  { event := event87364
    frameStart := 87267 },
  { event := event87365
    frameStart := 87267 },
  { event := event87366
    frameStart := 87267 },
  { event := event87367
    frameStart := 87267 },
  { event := event87368
    frameStart := 87267 },
  { event := event87369
    frameStart := 87267 },
  { event := event87370
    frameStart := 87267 },
  { event := event87371
    frameStart := 87267 },
  { event := event87372
    frameStart := 87267 },
  { event := event87373
    frameStart := 87267 },
  { event := event87374
    frameStart := 87267 },
  { event := event87375
    frameStart := 87267 }
]

def eventLeaf5461 : Array AnnotatedEvent := #[
  { event := event87376
    frameStart := 87267 },
  { event := event87377
    frameStart := 87267 },
  { event := event87378
    frameStart := 87267 },
  { event := event87379
    frameStart := 87267 },
  { event := event87380
    frameStart := 87267 },
  { event := event87381
    frameStart := 87267 },
  { event := event87382
    frameStart := 87267 },
  { event := event87383
    frameStart := 0 },
  { event := event87384
    frameStart := 0 },
  { event := event87385
    frameStart := 0 },
  { event := event87386
    frameStart := 0 },
  { event := event87387
    frameStart := 0 },
  { event := event87388
    frameStart := 0 },
  { event := event87389
    frameStart := 0 },
  { event := event87390
    frameStart := 0 },
  { event := event87391
    frameStart := 0 }
]

def eventLeaf5462 : Array AnnotatedEvent := #[
  { event := event87392
    frameStart := 0 },
  { event := event87393
    frameStart := 0 },
  { event := event87394
    frameStart := 0 },
  { event := event87395
    frameStart := 0 },
  { event := event87396
    frameStart := 0 },
  { event := event87397
    frameStart := 0 },
  { event := event87398
    frameStart := 0 },
  { event := event87399
    frameStart := 0 },
  { event := event87400
    frameStart := 0 },
  { event := event87401
    frameStart := 0 },
  { event := event87402
    frameStart := 0 },
  { event := event87403
    frameStart := 0 },
  { event := event87404
    frameStart := 0 },
  { event := event87405
    frameStart := 0 },
  { event := event87406
    frameStart := 0 },
  { event := event87407
    frameStart := 0 }
]

def eventLeaf5463 : Array AnnotatedEvent := #[
  { event := event87408
    frameStart := 0 },
  { event := event87409
    frameStart := 0 },
  { event := event87410
    frameStart := 0 },
  { event := event87411
    frameStart := 0 },
  { event := event87412
    frameStart := 0 },
  { event := event87413
    frameStart := 0 },
  { event := event87414
    frameStart := 0 },
  { event := event87415
    frameStart := 0 },
  { event := event87416
    frameStart := 0 },
  { event := event87417
    frameStart := 0 },
  { event := event87418
    frameStart := 0 },
  { event := event87419
    frameStart := 0 },
  { event := event87420
    frameStart := 87420 },
  { event := event87421
    frameStart := 87420 },
  { event := event87422
    frameStart := 87420 },
  { event := event87423
    frameStart := 87420 }
]

def eventLeaf5464 : Array AnnotatedEvent := #[
  { event := event87424
    frameStart := 87420 },
  { event := event87425
    frameStart := 87420 },
  { event := event87426
    frameStart := 87420 },
  { event := event87427
    frameStart := 87420 },
  { event := event87428
    frameStart := 87420 },
  { event := event87429
    frameStart := 87420 },
  { event := event87430
    frameStart := 87420 },
  { event := event87431
    frameStart := 87420 },
  { event := event87432
    frameStart := 87420 },
  { event := event87433
    frameStart := 87420 },
  { event := event87434
    frameStart := 87420 },
  { event := event87435
    frameStart := 87420 },
  { event := event87436
    frameStart := 87420 },
  { event := event87437
    frameStart := 87420 },
  { event := event87438
    frameStart := 87420 },
  { event := event87439
    frameStart := 87420 }
]

def eventLeaf5465 : Array AnnotatedEvent := #[
  { event := event87440
    frameStart := 87420 },
  { event := event87441
    frameStart := 87420 },
  { event := event87442
    frameStart := 87420 },
  { event := event87443
    frameStart := 87420 },
  { event := event87444
    frameStart := 87420 },
  { event := event87445
    frameStart := 87420 },
  { event := event87446
    frameStart := 87420 },
  { event := event87447
    frameStart := 87420 },
  { event := event87448
    frameStart := 87420 },
  { event := event87449
    frameStart := 87420 },
  { event := event87450
    frameStart := 87420 },
  { event := event87451
    frameStart := 87420 },
  { event := event87452
    frameStart := 87420 },
  { event := event87453
    frameStart := 87420 },
  { event := event87454
    frameStart := 87420 },
  { event := event87455
    frameStart := 87420 }
]

def eventLeaf5466 : Array AnnotatedEvent := #[
  { event := event87456
    frameStart := 87420 },
  { event := event87457
    frameStart := 87420 },
  { event := event87458
    frameStart := 87420 },
  { event := event87459
    frameStart := 87420 },
  { event := event87460
    frameStart := 87420 },
  { event := event87461
    frameStart := 87420 },
  { event := event87462
    frameStart := 87420 },
  { event := event87463
    frameStart := 87420 },
  { event := event87464
    frameStart := 87420 },
  { event := event87465
    frameStart := 87420 },
  { event := event87466
    frameStart := 87420 },
  { event := event87467
    frameStart := 87420 },
  { event := event87468
    frameStart := 87420 },
  { event := event87469
    frameStart := 87420 },
  { event := event87470
    frameStart := 87420 },
  { event := event87471
    frameStart := 87420 }
]

def eventLeaf5467 : Array AnnotatedEvent := #[
  { event := event87472
    frameStart := 87420 },
  { event := event87473
    frameStart := 87420 },
  { event := event87474
    frameStart := 87474 },
  { event := event87475
    frameStart := 87474 },
  { event := event87476
    frameStart := 87474 },
  { event := event87477
    frameStart := 87474 },
  { event := event87478
    frameStart := 87474 },
  { event := event87479
    frameStart := 87474 },
  { event := event87480
    frameStart := 87474 },
  { event := event87481
    frameStart := 87474 },
  { event := event87482
    frameStart := 87474 },
  { event := event87483
    frameStart := 87474 },
  { event := event87484
    frameStart := 87474 },
  { event := event87485
    frameStart := 87474 },
  { event := event87486
    frameStart := 87474 },
  { event := event87487
    frameStart := 87474 }
]

def eventLeaf5468 : Array AnnotatedEvent := #[
  { event := event87488
    frameStart := 87474 },
  { event := event87489
    frameStart := 87474 },
  { event := event87490
    frameStart := 87474 },
  { event := event87491
    frameStart := 87474 },
  { event := event87492
    frameStart := 87474 },
  { event := event87493
    frameStart := 87474 },
  { event := event87494
    frameStart := 87474 },
  { event := event87495
    frameStart := 87474 },
  { event := event87496
    frameStart := 87474 },
  { event := event87497
    frameStart := 87474 },
  { event := event87498
    frameStart := 87474 },
  { event := event87499
    frameStart := 87474 },
  { event := event87500
    frameStart := 87474 },
  { event := event87501
    frameStart := 87474 },
  { event := event87502
    frameStart := 87474 },
  { event := event87503
    frameStart := 87474 }
]

def eventLeaf5469 : Array AnnotatedEvent := #[
  { event := event87504
    frameStart := 87474 },
  { event := event87505
    frameStart := 87474 },
  { event := event87506
    frameStart := 87474 },
  { event := event87507
    frameStart := 87474 },
  { event := event87508
    frameStart := 87474 },
  { event := event87509
    frameStart := 87474 },
  { event := event87510
    frameStart := 87474 },
  { event := event87511
    frameStart := 87474 },
  { event := event87512
    frameStart := 87474 },
  { event := event87513
    frameStart := 87474 },
  { event := event87514
    frameStart := 87474 },
  { event := event87515
    frameStart := 87474 },
  { event := event87516
    frameStart := 87474 },
  { event := event87517
    frameStart := 87474 },
  { event := event87518
    frameStart := 87474 },
  { event := event87519
    frameStart := 87474 }
]

def eventLeaf5470 : Array AnnotatedEvent := #[
  { event := event87520
    frameStart := 87474 },
  { event := event87521
    frameStart := 87474 },
  { event := event87522
    frameStart := 87474 },
  { event := event87523
    frameStart := 87474 },
  { event := event87524
    frameStart := 87474 },
  { event := event87525
    frameStart := 87474 },
  { event := event87526
    frameStart := 87474 },
  { event := event87527
    frameStart := 87474 },
  { event := event87528
    frameStart := 87474 },
  { event := event87529
    frameStart := 87474 },
  { event := event87530
    frameStart := 87474 },
  { event := event87531
    frameStart := 87474 },
  { event := event87532
    frameStart := 87474 },
  { event := event87533
    frameStart := 87474 },
  { event := event87534
    frameStart := 87474 },
  { event := event87535
    frameStart := 87474 }
]

def eventLeaf5471 : Array AnnotatedEvent := #[
  { event := event87536
    frameStart := 87474 },
  { event := event87537
    frameStart := 87474 },
  { event := event87538
    frameStart := 87474 },
  { event := event87539
    frameStart := 87474 },
  { event := event87540
    frameStart := 87474 },
  { event := event87541
    frameStart := 87474 },
  { event := event87542
    frameStart := 87474 },
  { event := event87543
    frameStart := 87474 },
  { event := event87544
    frameStart := 87474 },
  { event := event87545
    frameStart := 87474 },
  { event := event87546
    frameStart := 87474 },
  { event := event87547
    frameStart := 87474 },
  { event := event87548
    frameStart := 87474 },
  { event := event87549
    frameStart := 87474 },
  { event := event87550
    frameStart := 87474 },
  { event := event87551
    frameStart := 87474 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events341
