import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events755

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event193280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47882⟩⟩) 0 ⟨5905⟩ 193279

def event193281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47882⟩⟩) (.authority (.programFamilyFact))

def exact193282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193282RawTermsValid :
    exact193282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47882⟩⟩) exact193282RawTerms (.finite 60) 193281 .exactZero (none)

def event193283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15111⟩⟩) 0 ⟨5905⟩ 193279

def event193284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15111⟩⟩) (.authority (.programFamilyFact))

def exact193285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩], []⟩, (1)⟩]

theorem exact193285RawTermsValid :
    exact193285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15111⟩⟩) exact193285RawTerms (.finite 60) 193284 .exactZero (none)

def event193286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 0 ⟨15111⟩ 193285

def event193287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 1 ⟨47882⟩ 193282

def event193288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.product (.predecessor 0 193286 .coefficient) (.predecessor 1 193287 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event193289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47883⟩⟩, .operator (⟨193285, 0⟩, ⟨193282, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩)

def exact193290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193290RawTermsValid :
    exact193290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47883⟩⟩) exact193290RawTerms (.finite 3600) 193288 .exactZero (none)

def event193291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47884⟩⟩) 0 ⟨47883⟩ 193290

def event193292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.identity (.predecessor 0 193291 .coefficient))

def event193293 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.finite 3600)

def event193294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48164⟩⟩) 0 ⟨47884⟩ 193293

def event193295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48164⟩⟩) (.authority (.programFamilyFact))

def exact193296RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], []⟩, (1)⟩]

theorem exact193296RawTermsValid :
    exact193296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48164⟩⟩) exact193296RawTerms (.finite 60) 193295 .exactZero (none)

def event193297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48165⟩⟩) 0 ⟨48164⟩ 193296

def event193298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48165⟩⟩) (.identity (.predecessor 0 193297 .coefficient))

def event193299 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48165⟩⟩) (.finite 60)

def event193300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49317⟩⟩) 0 ⟨48165⟩ 193299

def event193301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49317⟩⟩) (.authority (.programFamilyFact))

def event193302 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49317⟩⟩) (.finite 3720)

def event193303 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event193304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49319⟩⟩) 0 ⟨7177⟩ 193303

def event193305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49319⟩⟩) 1 ⟨49317⟩ 193302

def event193306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49319⟩⟩) (.authority (.operator))

def exact193307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩]

theorem exact193307RawTermsValid :
    exact193307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49319⟩⟩) exact193307RawTerms .large 193306 .exactZero (none)

def event193308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50079⟩⟩) 0 ⟨49319⟩ 193307

def event193309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50079⟩⟩) (.authority (.operator))

def exact193310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩]

theorem exact193310RawTermsValid :
    exact193310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50079⟩⟩) exact193310RawTerms (.finite 8192) 193309 .exactZero (none)

def event193311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event193312 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event193313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49514⟩⟩) 0 ⟨48165⟩ 193299

def event193314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49514⟩⟩) 1 ⟨136⟩ 193312

def event193315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49514⟩⟩) (.sum [.predecessor 0 193313 .coefficient, .predecessor 1 193314 .coefficient])

def event193316 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49514⟩⟩) (.finite 60)

def event193317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49515⟩⟩) 0 ⟨49514⟩ 193316

def event193318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49515⟩⟩) (.identity (.predecessor 0 193317 .coefficient))

def exact193319RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], []⟩, (1)⟩]

theorem exact193319RawTermsValid :
    exact193319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193319 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49515⟩⟩) exact193319RawTerms (.finite 60) 193318 .exactZero (none)

def event193320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact193321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193321RawTermsValid :
    exact193321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact193321RawTerms .large 193320 .exactZero (none)

def event193322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49516⟩⟩) 0 ⟨6908⟩ 193321

def event193323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49516⟩⟩) 1 ⟨49515⟩ 193319

def event193324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49516⟩⟩) (.product (.predecessor 0 193322 .coefficient) (.predecessor 1 193323 .coefficient) (⟨false, false, none, none, none⟩))

def event193325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49516⟩⟩, .operator (⟨193321, 0⟩, ⟨193319, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193326RawTermsValid :
    exact193326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49516⟩⟩) exact193326RawTerms .large 193324 .exactZero (none)

def event193327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 193303

def event193328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact193329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact193329RawTermsValid :
    exact193329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact193329RawTerms .large 193328 .exactZero (none)

def event193330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49517⟩⟩) 0 ⟨7196⟩ 193329

def event193331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49517⟩⟩) 1 ⟨49516⟩ 193326

def event193332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49517⟩⟩) (.sum [.predecessor 0 193330 .coefficient, .predecessor 1 193331 .coefficient])

def exact193333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193333RawTermsValid :
    exact193333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49517⟩⟩) exact193333RawTerms .large 193332 .exactZero (none)

def event193334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50080⟩⟩) 0 ⟨49517⟩ 193333

def event193335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50080⟩⟩) 1 ⟨50079⟩ 193310

def event193336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50080⟩⟩) (.product (.predecessor 0 193334 .coefficient) (.predecessor 1 193335 .coefficient) (⟨false, false, none, none, none⟩))

def event193337 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50080⟩⟩, .operator (⟨193333, 0⟩, ⟨193310, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩)

def event193338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50080⟩⟩, .operator (⟨193333, 1⟩, ⟨193310, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩)

def event193339 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50080⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50079⟩⟩) ⟨49319⟩ 193307)

def event193340 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50080⟩⟩, .relation 193339 0, ⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (-1)⟩)

def exact193341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (-1)⟩]

theorem exact193341RawTermsValid :
    exact193341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50080⟩⟩) exact193341RawTerms .large 193336 .exactZero (none)

def event193342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48389⟩⟩) 0 ⟨48165⟩ 193299

def event193343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48389⟩⟩) (.authority (.programFamilyFact))

def exact193344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], []⟩, (1)⟩]

theorem exact193344RawTermsValid :
    exact193344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48389⟩⟩) exact193344RawTerms (.finite 63) 193343 .exactZero (none)

def event193345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48390⟩⟩) 0 ⟨6908⟩ 193321

def event193346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48390⟩⟩) 1 ⟨48389⟩ 193344

def event193347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48390⟩⟩) (.product (.predecessor 0 193345 .coefficient) (.predecessor 1 193346 .coefficient) (⟨false, true, none, none, some 1⟩))

def event193348 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48390⟩⟩, .operator (⟨193321, 0⟩, ⟨193344, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193349RawTermsValid :
    exact193349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48390⟩⟩) exact193349RawTerms .large 193347 .exactZero (none)

def event193350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 193303

def event193351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact193352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact193352RawTermsValid :
    exact193352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact193352RawTerms .large 193351 .exactZero (none)

def event193353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48391⟩⟩) 0 ⟨7232⟩ 193352

def event193354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48391⟩⟩) 1 ⟨48390⟩ 193349

def event193355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48391⟩⟩) (.sum [.predecessor 0 193353 .coefficient, .predecessor 1 193354 .coefficient])

def exact193356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193356RawTermsValid :
    exact193356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48391⟩⟩) exact193356RawTerms .large 193355 .exactZero (none)

def event193357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50083⟩⟩) 0 ⟨48391⟩ 193356

def event193358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50083⟩⟩) 1 ⟨50080⟩ 193341

def event193359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50083⟩⟩) (.sum [.predecessor 0 193357 .coefficient, .predecessor 1 193358 .coefficient])

def exact193360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193360RawTermsValid :
    exact193360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50083⟩⟩) exact193360RawTerms .large 193359 .exactZero (none)

def event193361 : Event := .preFoldPolynomial 193360 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact193362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event193362 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50083⟩⟩) 193361 exact193362RawTerms .large 193359 .exactZero (none)

def event193363 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48165⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨193205, 193363⟩

def event193364 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48939⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩) (1) 0 2 (.universal 193363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩) (none) 193362)

def event193365 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48939⟩⟩, .relation 193364 1, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def event193366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48939⟩⟩, .relation 193364 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩)

def event193367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48939⟩⟩, .relation 193364 2, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩)

def event193368 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48939⟩⟩, .relation 193364 3, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact193369RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193369RawTermsValid :
    exact193369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48939⟩⟩) exact193369RawTerms .large 193201 (.finite 202072841853861888) (some (193203))

def event193370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50082⟩⟩) 0 ⟨48939⟩ 193369

def event193371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50082⟩⟩) 1 ⟨50081⟩ 193191

def event193372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50082⟩⟩) (.sum [.predecessor 0 193370 .coefficient, .predecessor 1 193371 .coefficient])

def event193373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50082⟩⟩, .operator (⟨193369, 0⟩, ⟨193191, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩)

def event193374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50082⟩⟩, .operator (⟨193369, 2⟩, ⟨193191, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (-1)⟩)

def event193375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50082⟩⟩) (.sum [.result 193369 .summary, .result 193191 .summary])

def exact193376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193376RawTermsValid :
    exact193376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50082⟩⟩) exact193376RawTerms .large 193372 (.finite 32194504275408640829496428331008) (some (193375))

def event193377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46637⟩⟩) 0 ⟨45485⟩ 9110

def event193378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46637⟩⟩) (.authority (.programFamilyFact))

def event193379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46637⟩⟩) (.finite 3720)

def event193380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46639⟩⟩) 0 ⟨7177⟩ 15500

def event193381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46639⟩⟩) 1 ⟨46637⟩ 193379

def event193382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46639⟩⟩) (.authority (.operator))

def exact193383RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46639⟩⟩]⟩, (1)⟩]

theorem exact193383RawTermsValid :
    exact193383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193383 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46639⟩⟩) exact193383RawTerms .large 193382 .exactZero (none)

def event193384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47399⟩⟩) 0 ⟨46639⟩ 193383

def event193385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47399⟩⟩) (.authority (.operator))

def exact193386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47399⟩⟩]⟩, (1)⟩]

theorem exact193386RawTermsValid :
    exact193386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47399⟩⟩) exact193386RawTerms (.finite 8192) 193385 .exactZero (none)

def event193387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46480⟩⟩) 0 ⟨45204⟩ 9104

def event193388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46480⟩⟩) (.authority (.programFamilyFact))

def event193389 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46480⟩⟩) (.finite 3720)

def event193390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46481⟩⟩) 0 ⟨7177⟩ 15500

def event193391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46481⟩⟩) 1 ⟨46480⟩ 193389

def event193392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46481⟩⟩) (.authority (.operator))

def exact193393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩, (1)⟩]

theorem exact193393RawTermsValid :
    exact193393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46481⟩⟩) exact193393RawTerms .large 193392 .exactZero (none)

def event193394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47001⟩⟩) 0 ⟨46481⟩ 193393

def event193395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47001⟩⟩) (.authority (.operator))

def exact193396RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩, (1)⟩]

theorem exact193396RawTermsValid :
    exact193396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47001⟩⟩) exact193396RawTerms (.finite 8192) 193395 .exactZero (none)

def event193397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45205⟩⟩) 0 ⟨45202⟩ 9093

def event193398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45205⟩⟩) 1 ⟨6998⟩ 192903

def event193399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45205⟩⟩) (.tensor (.predecessor 0 193397 .coefficient) (.predecessor 1 193398 .coefficient) true false)

def event193400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45205⟩⟩, .operator (⟨9093, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193401RawTermsValid :
    exact193401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45205⟩⟩) exact193401RawTerms .large 193399 .exactZero (none)

def event193402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8818⟩⟩) 0 ⟨5907⟩ 192773

def event193403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8818⟩⟩) 1 ⟨7284⟩ 17581

def event193404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8818⟩⟩) (.product (.predecessor 0 193402 .coefficient) (.predecessor 1 193403 .coefficient) (⟨false, false, none, none, none⟩))

def event193405 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8818⟩⟩, .operator (⟨192773, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact193406RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact193406RawTermsValid :
    exact193406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193406 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8818⟩⟩) exact193406RawTerms .large 193404 .exactZero (none)

def event193407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45206⟩⟩) 0 ⟨8818⟩ 193406

def event193408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45206⟩⟩) 1 ⟨45205⟩ 193401

def event193409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45206⟩⟩) (.sum [.predecessor 0 193407 .coefficient, .predecessor 1 193408 .coefficient])

def exact193410RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193410RawTermsValid :
    exact193410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193410 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45206⟩⟩) exact193410RawTerms .large 193409 .exactZero (none)

def event193411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45207⟩⟩) 0 ⟨45206⟩ 193410

def event193412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45207⟩⟩) 1 ⟨110⟩ 17573

def event193413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45207⟩⟩) (.sum [.predecessor 0 193411 .coefficient, .predecessor 1 193412 .coefficient])

def event193414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45207⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩) [⟨.result 17573 .coefficient, false, none⟩])

def event193415 : Event := .survivorFold (1) 193414

def exact193416RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193416RawTermsValid :
    exact193416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45207⟩⟩) exact193416RawTerms .large 193413 (.finite 26) (some (193414))

def event193417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45208⟩⟩) 0 ⟨45207⟩ 193416

def event193418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45208⟩⟩) 1 ⟨14811⟩ 9096

def event193419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45208⟩⟩) (.product (.predecessor 0 193417 .coefficient) (.predecessor 1 193418 .coefficient) (⟨false, true, none, none, some 1⟩))

def event193420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45208⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩], []⟩) [⟨.result 9096 .coefficient, true, some 1⟩])

def event193421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45208⟩⟩) (.product (.result 193416 .summary) (.transfer 193420) (⟨false, false, none, none, none⟩))

def event193422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45208⟩⟩, .operator (⟨193416, 1⟩, ⟨9096, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event193423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45208⟩⟩, .operator (⟨193416, 0⟩, ⟨9096, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact193424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193424RawTermsValid :
    exact193424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45208⟩⟩) exact193424RawTerms .large 193419 (.finite 49414144) (some (193421))

def event193425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14812⟩⟩) 0 ⟨14811⟩ 9096

def event193426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14812⟩⟩) 1 ⟨6998⟩ 192903

def event193427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14812⟩⟩) (.tensor (.predecessor 0 193425 .coefficient) (.predecessor 1 193426 .coefficient) true false)

def event193428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14812⟩⟩, .operator (⟨9096, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193429RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193429RawTermsValid :
    exact193429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193429 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14812⟩⟩) exact193429RawTerms .large 193427 .exactZero (none)

def event193430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8835⟩⟩) 0 ⟨5907⟩ 192773

def event193431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8835⟩⟩) 1 ⟨7301⟩ 17622

def event193432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8835⟩⟩) (.product (.predecessor 0 193430 .coefficient) (.predecessor 1 193431 .coefficient) (⟨false, false, none, none, none⟩))

def event193433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8835⟩⟩, .operator (⟨192773, 0⟩, ⟨17622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩)

def exact193434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact193434RawTermsValid :
    exact193434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8835⟩⟩) exact193434RawTerms .large 193432 .exactZero (none)

def event193435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14813⟩⟩) 0 ⟨8835⟩ 193434

def event193436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14813⟩⟩) 1 ⟨14812⟩ 193429

def event193437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14813⟩⟩) (.sum [.predecessor 0 193435 .coefficient, .predecessor 1 193436 .coefficient])

def exact193438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193438RawTermsValid :
    exact193438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14813⟩⟩) exact193438RawTerms .large 193437 .exactZero (none)

def event193439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14814⟩⟩) 0 ⟨14813⟩ 193438

def event193440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14814⟩⟩) 1 ⟨127⟩ 17614

def event193441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14814⟩⟩) (.sum [.predecessor 0 193439 .coefficient, .predecessor 1 193440 .coefficient])

def event193442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14814⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩) [⟨.result 17614 .coefficient, false, none⟩])

def event193443 : Event := .survivorFold (1) 193442

def exact193444RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193444RawTermsValid :
    exact193444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14814⟩⟩) exact193444RawTerms .large 193441 (.finite 26) (some (193442))

def event193445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14815⟩⟩) 0 ⟨14814⟩ 193444

def event193446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14815⟩⟩) 1 ⟨9563⟩ 17611

def event193447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14815⟩⟩) (.product (.predecessor 0 193445 .coefficient) (.predecessor 1 193446 .coefficient) (⟨false, false, none, none, none⟩))

def event193448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14815⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) [⟨.result 17607 .coefficient, false, none⟩])

def event193449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14815⟩⟩) (.product (.result 193444 .summary) (.transfer 193448) (⟨false, false, none, none, none⟩))

def event193450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14815⟩⟩, .operator (⟨193444, 1⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (-1)⟩)

def event193451 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14815⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581)

def event193452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14815⟩⟩, .relation 193451 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩)

def event193453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14815⟩⟩, .operator (⟨193444, 0⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact193454RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩]

theorem exact193454RawTermsValid :
    exact193454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14815⟩⟩) exact193454RawTerms .large 193447 (.finite 279172874240) (some (193449))

def event193455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45209⟩⟩) 0 ⟨14815⟩ 193454

def event193456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45209⟩⟩) 1 ⟨45208⟩ 193424

def event193457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45209⟩⟩) (.sum [.predecessor 0 193455 .coefficient, .predecessor 1 193456 .coefficient])

def event193458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45209⟩⟩, .operator (⟨193454, 1⟩, ⟨193424, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def event193459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45209⟩⟩) (.sum [.result 193454 .summary, .result 193424 .summary])

def exact193460RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193460RawTermsValid :
    exact193460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45209⟩⟩) exact193460RawTerms .large 193457 (.finite 279222288384) (some (193459))

def event193461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47002⟩⟩) 0 ⟨45209⟩ 193460

def event193462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47002⟩⟩) 1 ⟨47001⟩ 193396

def event193463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47002⟩⟩) (.product (.predecessor 0 193461 .coefficient) (.predecessor 1 193462 .coefficient) (⟨false, false, none, none, none⟩))

def event193464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47002⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) [⟨.result 193396 .coefficient, false, none⟩])

def event193465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47002⟩⟩) (.product (.result 193460 .summary) (.transfer 193464) (⟨false, false, none, none, none⟩))

def event193466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47002⟩⟩, .operator (⟨193460, 1⟩, ⟨193396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩, (-1)⟩)

def event193467 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47002⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47001⟩⟩) ⟨46481⟩ 193393)

def event193468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47002⟩⟩, .relation 193467 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩, (-1)⟩)

def event193469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47002⟩⟩, .operator (⟨193460, 0⟩, ⟨193396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩, (1)⟩)

def exact193470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩, (-1)⟩]

theorem exact193470RawTermsValid :
    exact193470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47002⟩⟩) exact193470RawTerms .large 193463 (.finite 2998126492308901724160) (some (193465))

def event193471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45929⟩⟩) 0 ⟨45204⟩ 9104

def event193472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45929⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact193473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩]

theorem exact193473RawTermsValid :
    exact193473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45929⟩⟩) exact193473RawTerms (.finite 5647228698) 193472 .exactZero (none)

def event193474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45931⟩⟩) 0 ⟨45929⟩ 193473

def event193475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45931⟩⟩) 1 ⟨2370⟩ 4

def event193476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45931⟩⟩) (.scale (.predecessor 0 193474 .coefficient) (.value (.predecessor 1 193475 .coefficient)))

def exact193477RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩]

theorem exact193477RawTermsValid :
    exact193477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193477 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45931⟩⟩) exact193477RawTerms (.finite 5647228698) 193476 .exactZero (none)

def event193478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45932⟩⟩) 0 ⟨5909⟩ 192995

def event193479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45932⟩⟩) 1 ⟨45931⟩ 193477

def event193480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45932⟩⟩) (.product (.predecessor 0 193478 .coefficient) (.predecessor 1 193479 .coefficient) (⟨false, false, none, none, none⟩))

def event193481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45932⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) [⟨.result 193473 .coefficient, false, none⟩])

def event193482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45932⟩⟩) (.product (.result 192995 .summary) (.transfer 193481) (⟨false, false, none, none, none⟩))

def event193483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45932⟩⟩, .operator (⟨192995, 0⟩, ⟨193477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩)

def event193484 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨45930⟩⟩)

def event193485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193486 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event193488 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event193489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event193490 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event193491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event193492 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event193493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 193492

def event193494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 193490

def event193495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 193493 .coefficient) (.value (.predecessor 1 193494 .coefficient)))

def event193496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event193497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 193496

def event193498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 193488

def event193499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 193497 .coefficient, .predecessor 1 193498 .coefficient])

def event193500 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event193501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 193500

def event193502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 193486

def event193503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 193502 .coefficient))

def event193504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event193505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45202⟩⟩) 0 ⟨5905⟩ 193504

def event193506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45202⟩⟩) (.authority (.programFamilyFact))

def exact193507RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45202⟩⟩], []⟩, (1)⟩]

theorem exact193507RawTermsValid :
    exact193507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45202⟩⟩) exact193507RawTerms (.finite 58) 193506 .exactZero (none)

def event193508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14811⟩⟩) 0 ⟨5905⟩ 193504

def event193509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14811⟩⟩) (.authority (.programFamilyFact))

def exact193510RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩], []⟩, (1)⟩]

theorem exact193510RawTermsValid :
    exact193510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14811⟩⟩) exact193510RawTerms (.finite 58) 193509 .exactZero (none)

def event193511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45203⟩⟩) 0 ⟨14811⟩ 193510

def event193512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45203⟩⟩) 1 ⟨45202⟩ 193507

def event193513 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45203⟩⟩) (.product (.predecessor 0 193511 .coefficient) (.predecessor 1 193512 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event193514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45203⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], []⟩) [⟨.result 193510 .coefficient, true, some 1⟩, ⟨.result 193507 .coefficient, true, some 1⟩])

def event193515 : Event := .survivorFold (1) 193514

def exact193516RawTerms : List Term := []

theorem exact193516RawTermsValid :
    exact193516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45203⟩⟩) exact193516RawTerms (.finite 3364) 193513 (.finite 3364) (some (193514))

def event193517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45204⟩⟩) 0 ⟨45203⟩ 193516

def event193518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45204⟩⟩) (.identity (.predecessor 0 193517 .coefficient))

def event193519 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45204⟩⟩) (.finite 3364)

def event193520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45929⟩⟩) 0 ⟨45204⟩ 193519

def event193521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45929⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact193522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩]

theorem exact193522RawTermsValid :
    exact193522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45929⟩⟩) exact193522RawTerms (.finite 5647228698) 193521 .exactZero (none)

def event193523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact193524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact193524RawTermsValid :
    exact193524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact193524RawTerms .large 193523 .exactZero (none)

def event193525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45930⟩⟩) 0 ⟨35⟩ 193524

def event193526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45930⟩⟩) 1 ⟨45929⟩ 193522

def event193527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45930⟩⟩) (.product (.predecessor 0 193525 .coefficient) (.predecessor 1 193526 .coefficient) (⟨false, false, none, none, none⟩))

def event193528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45930⟩⟩, .operator (⟨193524, 0⟩, ⟨193522, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩)

def exact193529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩]

theorem exact193529RawTermsValid :
    exact193529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45930⟩⟩) exact193529RawTerms .large 193527 .exactZero (none)

def event193530 : Event := .preFoldPolynomial 193529 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩] .exactZero none

def exact193531RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩, (1)⟩]

def event193531 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨45930⟩⟩) 193530 exact193531RawTerms .large 193527 .exactZero (none)

def event193532 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47005⟩⟩)

def event193533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def eventLeaf12080 : Array AnnotatedEvent := #[
  { event := event193280
    frameStart := 193259 },
  { event := event193281
    frameStart := 193259 },
  { event := event193282
    frameStart := 193259 },
  { event := event193283
    frameStart := 193259 },
  { event := event193284
    frameStart := 193259 },
  { event := event193285
    frameStart := 193259 },
  { event := event193286
    frameStart := 193259 },
  { event := event193287
    frameStart := 193259 },
  { event := event193288
    frameStart := 193259 },
  { event := event193289
    frameStart := 193259 },
  { event := event193290
    frameStart := 193259 },
  { event := event193291
    frameStart := 193259 },
  { event := event193292
    frameStart := 193259 },
  { event := event193293
    frameStart := 193259 },
  { event := event193294
    frameStart := 193259 },
  { event := event193295
    frameStart := 193259 }
]

def eventLeaf12081 : Array AnnotatedEvent := #[
  { event := event193296
    frameStart := 193259 },
  { event := event193297
    frameStart := 193259 },
  { event := event193298
    frameStart := 193259 },
  { event := event193299
    frameStart := 193259 },
  { event := event193300
    frameStart := 193259 },
  { event := event193301
    frameStart := 193259 },
  { event := event193302
    frameStart := 193259 },
  { event := event193303
    frameStart := 193259 },
  { event := event193304
    frameStart := 193259 },
  { event := event193305
    frameStart := 193259 },
  { event := event193306
    frameStart := 193259 },
  { event := event193307
    frameStart := 193259 },
  { event := event193308
    frameStart := 193259 },
  { event := event193309
    frameStart := 193259 },
  { event := event193310
    frameStart := 193259 },
  { event := event193311
    frameStart := 193259 }
]

def eventLeaf12082 : Array AnnotatedEvent := #[
  { event := event193312
    frameStart := 193259 },
  { event := event193313
    frameStart := 193259 },
  { event := event193314
    frameStart := 193259 },
  { event := event193315
    frameStart := 193259 },
  { event := event193316
    frameStart := 193259 },
  { event := event193317
    frameStart := 193259 },
  { event := event193318
    frameStart := 193259 },
  { event := event193319
    frameStart := 193259 },
  { event := event193320
    frameStart := 193259 },
  { event := event193321
    frameStart := 193259 },
  { event := event193322
    frameStart := 193259 },
  { event := event193323
    frameStart := 193259 },
  { event := event193324
    frameStart := 193259 },
  { event := event193325
    frameStart := 193259 },
  { event := event193326
    frameStart := 193259 },
  { event := event193327
    frameStart := 193259 }
]

def eventLeaf12083 : Array AnnotatedEvent := #[
  { event := event193328
    frameStart := 193259 },
  { event := event193329
    frameStart := 193259 },
  { event := event193330
    frameStart := 193259 },
  { event := event193331
    frameStart := 193259 },
  { event := event193332
    frameStart := 193259 },
  { event := event193333
    frameStart := 193259 },
  { event := event193334
    frameStart := 193259 },
  { event := event193335
    frameStart := 193259 },
  { event := event193336
    frameStart := 193259 },
  { event := event193337
    frameStart := 193259 },
  { event := event193338
    frameStart := 193259 },
  { event := event193339
    frameStart := 193259 },
  { event := event193340
    frameStart := 193259 },
  { event := event193341
    frameStart := 193259 },
  { event := event193342
    frameStart := 193259 },
  { event := event193343
    frameStart := 193259 }
]

def eventLeaf12084 : Array AnnotatedEvent := #[
  { event := event193344
    frameStart := 193259 },
  { event := event193345
    frameStart := 193259 },
  { event := event193346
    frameStart := 193259 },
  { event := event193347
    frameStart := 193259 },
  { event := event193348
    frameStart := 193259 },
  { event := event193349
    frameStart := 193259 },
  { event := event193350
    frameStart := 193259 },
  { event := event193351
    frameStart := 193259 },
  { event := event193352
    frameStart := 193259 },
  { event := event193353
    frameStart := 193259 },
  { event := event193354
    frameStart := 193259 },
  { event := event193355
    frameStart := 193259 },
  { event := event193356
    frameStart := 193259 },
  { event := event193357
    frameStart := 193259 },
  { event := event193358
    frameStart := 193259 },
  { event := event193359
    frameStart := 193259 }
]

def eventLeaf12085 : Array AnnotatedEvent := #[
  { event := event193360
    frameStart := 193259 },
  { event := event193361
    frameStart := 193259 },
  { event := event193362
    frameStart := 193259 },
  { event := event193363
    frameStart := 0 },
  { event := event193364
    frameStart := 0 },
  { event := event193365
    frameStart := 0 },
  { event := event193366
    frameStart := 0 },
  { event := event193367
    frameStart := 0 },
  { event := event193368
    frameStart := 0 },
  { event := event193369
    frameStart := 0 },
  { event := event193370
    frameStart := 0 },
  { event := event193371
    frameStart := 0 },
  { event := event193372
    frameStart := 0 },
  { event := event193373
    frameStart := 0 },
  { event := event193374
    frameStart := 0 },
  { event := event193375
    frameStart := 0 }
]

def eventLeaf12086 : Array AnnotatedEvent := #[
  { event := event193376
    frameStart := 0 },
  { event := event193377
    frameStart := 0 },
  { event := event193378
    frameStart := 0 },
  { event := event193379
    frameStart := 0 },
  { event := event193380
    frameStart := 0 },
  { event := event193381
    frameStart := 0 },
  { event := event193382
    frameStart := 0 },
  { event := event193383
    frameStart := 0 },
  { event := event193384
    frameStart := 0 },
  { event := event193385
    frameStart := 0 },
  { event := event193386
    frameStart := 0 },
  { event := event193387
    frameStart := 0 },
  { event := event193388
    frameStart := 0 },
  { event := event193389
    frameStart := 0 },
  { event := event193390
    frameStart := 0 },
  { event := event193391
    frameStart := 0 }
]

def eventLeaf12087 : Array AnnotatedEvent := #[
  { event := event193392
    frameStart := 0 },
  { event := event193393
    frameStart := 0 },
  { event := event193394
    frameStart := 0 },
  { event := event193395
    frameStart := 0 },
  { event := event193396
    frameStart := 0 },
  { event := event193397
    frameStart := 0 },
  { event := event193398
    frameStart := 0 },
  { event := event193399
    frameStart := 0 },
  { event := event193400
    frameStart := 0 },
  { event := event193401
    frameStart := 0 },
  { event := event193402
    frameStart := 0 },
  { event := event193403
    frameStart := 0 },
  { event := event193404
    frameStart := 0 },
  { event := event193405
    frameStart := 0 },
  { event := event193406
    frameStart := 0 },
  { event := event193407
    frameStart := 0 }
]

def eventLeaf12088 : Array AnnotatedEvent := #[
  { event := event193408
    frameStart := 0 },
  { event := event193409
    frameStart := 0 },
  { event := event193410
    frameStart := 0 },
  { event := event193411
    frameStart := 0 },
  { event := event193412
    frameStart := 0 },
  { event := event193413
    frameStart := 0 },
  { event := event193414
    frameStart := 0 },
  { event := event193415
    frameStart := 0 },
  { event := event193416
    frameStart := 0 },
  { event := event193417
    frameStart := 0 },
  { event := event193418
    frameStart := 0 },
  { event := event193419
    frameStart := 0 },
  { event := event193420
    frameStart := 0 },
  { event := event193421
    frameStart := 0 },
  { event := event193422
    frameStart := 0 },
  { event := event193423
    frameStart := 0 }
]

def eventLeaf12089 : Array AnnotatedEvent := #[
  { event := event193424
    frameStart := 0 },
  { event := event193425
    frameStart := 0 },
  { event := event193426
    frameStart := 0 },
  { event := event193427
    frameStart := 0 },
  { event := event193428
    frameStart := 0 },
  { event := event193429
    frameStart := 0 },
  { event := event193430
    frameStart := 0 },
  { event := event193431
    frameStart := 0 },
  { event := event193432
    frameStart := 0 },
  { event := event193433
    frameStart := 0 },
  { event := event193434
    frameStart := 0 },
  { event := event193435
    frameStart := 0 },
  { event := event193436
    frameStart := 0 },
  { event := event193437
    frameStart := 0 },
  { event := event193438
    frameStart := 0 },
  { event := event193439
    frameStart := 0 }
]

def eventLeaf12090 : Array AnnotatedEvent := #[
  { event := event193440
    frameStart := 0 },
  { event := event193441
    frameStart := 0 },
  { event := event193442
    frameStart := 0 },
  { event := event193443
    frameStart := 0 },
  { event := event193444
    frameStart := 0 },
  { event := event193445
    frameStart := 0 },
  { event := event193446
    frameStart := 0 },
  { event := event193447
    frameStart := 0 },
  { event := event193448
    frameStart := 0 },
  { event := event193449
    frameStart := 0 },
  { event := event193450
    frameStart := 0 },
  { event := event193451
    frameStart := 0 },
  { event := event193452
    frameStart := 0 },
  { event := event193453
    frameStart := 0 },
  { event := event193454
    frameStart := 0 },
  { event := event193455
    frameStart := 0 }
]

def eventLeaf12091 : Array AnnotatedEvent := #[
  { event := event193456
    frameStart := 0 },
  { event := event193457
    frameStart := 0 },
  { event := event193458
    frameStart := 0 },
  { event := event193459
    frameStart := 0 },
  { event := event193460
    frameStart := 0 },
  { event := event193461
    frameStart := 0 },
  { event := event193462
    frameStart := 0 },
  { event := event193463
    frameStart := 0 },
  { event := event193464
    frameStart := 0 },
  { event := event193465
    frameStart := 0 },
  { event := event193466
    frameStart := 0 },
  { event := event193467
    frameStart := 0 },
  { event := event193468
    frameStart := 0 },
  { event := event193469
    frameStart := 0 },
  { event := event193470
    frameStart := 0 },
  { event := event193471
    frameStart := 0 }
]

def eventLeaf12092 : Array AnnotatedEvent := #[
  { event := event193472
    frameStart := 0 },
  { event := event193473
    frameStart := 0 },
  { event := event193474
    frameStart := 0 },
  { event := event193475
    frameStart := 0 },
  { event := event193476
    frameStart := 0 },
  { event := event193477
    frameStart := 0 },
  { event := event193478
    frameStart := 0 },
  { event := event193479
    frameStart := 0 },
  { event := event193480
    frameStart := 0 },
  { event := event193481
    frameStart := 0 },
  { event := event193482
    frameStart := 0 },
  { event := event193483
    frameStart := 0 },
  { event := event193484
    frameStart := 193484 },
  { event := event193485
    frameStart := 193484 },
  { event := event193486
    frameStart := 193484 },
  { event := event193487
    frameStart := 193484 }
]

def eventLeaf12093 : Array AnnotatedEvent := #[
  { event := event193488
    frameStart := 193484 },
  { event := event193489
    frameStart := 193484 },
  { event := event193490
    frameStart := 193484 },
  { event := event193491
    frameStart := 193484 },
  { event := event193492
    frameStart := 193484 },
  { event := event193493
    frameStart := 193484 },
  { event := event193494
    frameStart := 193484 },
  { event := event193495
    frameStart := 193484 },
  { event := event193496
    frameStart := 193484 },
  { event := event193497
    frameStart := 193484 },
  { event := event193498
    frameStart := 193484 },
  { event := event193499
    frameStart := 193484 },
  { event := event193500
    frameStart := 193484 },
  { event := event193501
    frameStart := 193484 },
  { event := event193502
    frameStart := 193484 },
  { event := event193503
    frameStart := 193484 }
]

def eventLeaf12094 : Array AnnotatedEvent := #[
  { event := event193504
    frameStart := 193484 },
  { event := event193505
    frameStart := 193484 },
  { event := event193506
    frameStart := 193484 },
  { event := event193507
    frameStart := 193484 },
  { event := event193508
    frameStart := 193484 },
  { event := event193509
    frameStart := 193484 },
  { event := event193510
    frameStart := 193484 },
  { event := event193511
    frameStart := 193484 },
  { event := event193512
    frameStart := 193484 },
  { event := event193513
    frameStart := 193484 },
  { event := event193514
    frameStart := 193484 },
  { event := event193515
    frameStart := 193484 },
  { event := event193516
    frameStart := 193484 },
  { event := event193517
    frameStart := 193484 },
  { event := event193518
    frameStart := 193484 },
  { event := event193519
    frameStart := 193484 }
]

def eventLeaf12095 : Array AnnotatedEvent := #[
  { event := event193520
    frameStart := 193484 },
  { event := event193521
    frameStart := 193484 },
  { event := event193522
    frameStart := 193484 },
  { event := event193523
    frameStart := 193484 },
  { event := event193524
    frameStart := 193484 },
  { event := event193525
    frameStart := 193484 },
  { event := event193526
    frameStart := 193484 },
  { event := event193527
    frameStart := 193484 },
  { event := event193528
    frameStart := 193484 },
  { event := event193529
    frameStart := 193484 },
  { event := event193530
    frameStart := 193484 },
  { event := event193531
    frameStart := 193484 },
  { event := event193532
    frameStart := 193532 },
  { event := event193533
    frameStart := 193532 },
  { event := event193534
    frameStart := 193532 },
  { event := event193535
    frameStart := 193532 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events755
