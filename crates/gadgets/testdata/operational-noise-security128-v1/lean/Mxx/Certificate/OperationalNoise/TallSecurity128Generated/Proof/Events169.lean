import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events169

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event43264 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event43265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43266

def event43268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 43264

def event43269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43267 .coefficient) (.value (.predecessor 1 43268 .coefficient)))

def event43270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43270

def event43272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 43262

def event43273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43271 .coefficient, .predecessor 1 43272 .coefficient])

def event43274 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43274

def event43276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 43260

def event43277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43276 .coefficient))

def event43278 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37330⟩⟩) 0 ⟨11600⟩ 43278

def event43280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37330⟩⟩) (.authority (.programFamilyFact))

def exact43281RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩, (1)⟩]

theorem exact43281RawTermsValid :
    exact43281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43281 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37330⟩⟩) exact43281RawTerms (.finite 42) 43280 .exactZero (none)

def event43282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14016⟩⟩) 0 ⟨11600⟩ 43278

def event43283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14016⟩⟩) (.authority (.programFamilyFact))

def exact43284RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩], []⟩, (1)⟩]

theorem exact43284RawTermsValid :
    exact43284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14016⟩⟩) exact43284RawTerms (.finite 42) 43283 .exactZero (none)

def event43285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37331⟩⟩) 0 ⟨14016⟩ 43284

def event43286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37331⟩⟩) 1 ⟨37330⟩ 43281

def event43287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37331⟩⟩) (.product (.predecessor 0 43285 .coefficient) (.predecessor 1 43286 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37331⟩⟩, .operator (⟨43284, 0⟩, ⟨43281, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩, (1)⟩)

def exact43289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩, (1)⟩]

theorem exact43289RawTermsValid :
    exact43289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37331⟩⟩) exact43289RawTerms (.finite 1764) 43287 .exactZero (none)

def event43290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37332⟩⟩) 0 ⟨37331⟩ 43289

def event43291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37332⟩⟩) (.identity (.predecessor 0 43290 .coefficient))

def event43292 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37332⟩⟩) (.finite 1764)

def event43293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37500⟩⟩) 0 ⟨37332⟩ 43292

def event43294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37500⟩⟩) (.authority (.programFamilyFact))

def exact43295RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], []⟩, (1)⟩]

theorem exact43295RawTermsValid :
    exact43295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37500⟩⟩) exact43295RawTerms (.finite 42) 43294 .exactZero (none)

def event43296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37501⟩⟩) 0 ⟨37500⟩ 43295

def event43297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37501⟩⟩) (.identity (.predecessor 0 43296 .coefficient))

def event43298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37501⟩⟩) (.finite 42)

def event43299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38660⟩⟩) 0 ⟨37501⟩ 43298

def event43300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38660⟩⟩) (.authority (.programFamilyFact))

def event43301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38660⟩⟩) (.finite 3720)

def event43302 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event43303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38661⟩⟩) 0 ⟨7177⟩ 43302

def event43304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38661⟩⟩) 1 ⟨38660⟩ 43301

def event43305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38661⟩⟩) (.authority (.operator))

def exact43306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩]

theorem exact43306RawTermsValid :
    exact43306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38661⟩⟩) exact43306RawTerms .large 43305 .exactZero (none)

def event43307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39528⟩⟩) 0 ⟨38661⟩ 43306

def event43308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39528⟩⟩) (.authority (.operator))

def exact43309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩]

theorem exact43309RawTermsValid :
    exact43309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39528⟩⟩) exact43309RawTerms (.finite 8192) 43308 .exactZero (none)

def event43310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event43311 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event43312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38822⟩⟩) 0 ⟨37501⟩ 43298

def event43313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38822⟩⟩) 1 ⟨136⟩ 43311

def event43314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38822⟩⟩) (.sum [.predecessor 0 43312 .coefficient, .predecessor 1 43313 .coefficient])

def event43315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38822⟩⟩) (.finite 42)

def event43316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38823⟩⟩) 0 ⟨38822⟩ 43315

def event43317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38823⟩⟩) (.identity (.predecessor 0 43316 .coefficient))

def exact43318RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], []⟩, (1)⟩]

theorem exact43318RawTermsValid :
    exact43318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38823⟩⟩) exact43318RawTerms (.finite 42) 43317 .exactZero (none)

def event43319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact43320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43320RawTermsValid :
    exact43320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact43320RawTerms .large 43319 .exactZero (none)

def event43321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38824⟩⟩) 0 ⟨6908⟩ 43320

def event43322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38824⟩⟩) 1 ⟨38823⟩ 43318

def event43323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38824⟩⟩) (.product (.predecessor 0 43321 .coefficient) (.predecessor 1 43322 .coefficient) (⟨false, false, none, none, none⟩))

def event43324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38824⟩⟩, .operator (⟨43320, 0⟩, ⟨43318, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact43325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43325RawTermsValid :
    exact43325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38824⟩⟩) exact43325RawTerms .large 43323 .exactZero (none)

def event43326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 43302

def event43327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact43328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact43328RawTermsValid :
    exact43328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43328 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact43328RawTerms .large 43327 .exactZero (none)

def event43329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38825⟩⟩) 0 ⟨7192⟩ 43328

def event43330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38825⟩⟩) 1 ⟨38824⟩ 43325

def event43331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38825⟩⟩) (.sum [.predecessor 0 43329 .coefficient, .predecessor 1 43330 .coefficient])

def exact43332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43332RawTermsValid :
    exact43332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38825⟩⟩) exact43332RawTerms .large 43331 .exactZero (none)

def event43333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39529⟩⟩) 0 ⟨38825⟩ 43332

def event43334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39529⟩⟩) 1 ⟨39528⟩ 43309

def event43335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39529⟩⟩) (.product (.predecessor 0 43333 .coefficient) (.predecessor 1 43334 .coefficient) (⟨false, false, none, none, none⟩))

def event43336 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39529⟩⟩, .operator (⟨43332, 0⟩, ⟨43309, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩)

def event43337 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39529⟩⟩, .operator (⟨43332, 1⟩, ⟨43309, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩)

def event43338 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39529⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39528⟩⟩) ⟨38661⟩ 43306)

def event43339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39529⟩⟩, .relation 43338 0, ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (-1)⟩)

def exact43340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (-1)⟩]

theorem exact43340RawTermsValid :
    exact43340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39529⟩⟩) exact43340RawTerms .large 43335 .exactZero (none)

def event43341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37756⟩⟩) 0 ⟨37501⟩ 43298

def event43342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37756⟩⟩) (.authority (.programFamilyFact))

def exact43343RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], []⟩, (1)⟩]

theorem exact43343RawTermsValid :
    exact43343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37756⟩⟩) exact43343RawTerms (.finite 42) 43342 .exactZero (none)

def event43344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37758⟩⟩) 0 ⟨6908⟩ 43320

def event43345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37758⟩⟩) 1 ⟨37756⟩ 43343

def event43346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37758⟩⟩) (.product (.predecessor 0 43344 .coefficient) (.predecessor 1 43345 .coefficient) (⟨false, true, none, none, some 1⟩))

def event43347 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37758⟩⟩, .operator (⟨43320, 0⟩, ⟨43343, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact43348RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact43348RawTermsValid :
    exact43348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37758⟩⟩) exact43348RawTerms .large 43346 .exactZero (none)

def event43349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7223⟩⟩) 0 ⟨7177⟩ 43302

def event43350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7223⟩⟩) (.authority (.operator))

def exact43351RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩]

theorem exact43351RawTermsValid :
    exact43351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7223⟩⟩) exact43351RawTerms .large 43350 .exactZero (none)

def event43352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37759⟩⟩) 0 ⟨7223⟩ 43351

def event43353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37759⟩⟩) 1 ⟨37758⟩ 43348

def event43354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37759⟩⟩) (.sum [.predecessor 0 43352 .coefficient, .predecessor 1 43353 .coefficient])

def exact43355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43355RawTermsValid :
    exact43355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37759⟩⟩) exact43355RawTerms .large 43354 .exactZero (none)

def event43356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39533⟩⟩) 0 ⟨37759⟩ 43355

def event43357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39533⟩⟩) 1 ⟨39529⟩ 43340

def event43358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39533⟩⟩) (.sum [.predecessor 0 43356 .coefficient, .predecessor 1 43357 .coefficient])

def exact43359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43359RawTermsValid :
    exact43359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39533⟩⟩) exact43359RawTerms .large 43358 .exactZero (none)

def event43360 : Event := .preFoldPolynomial 43359 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact43361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event43361 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39533⟩⟩) 43360 exact43361RawTerms .large 43358 .exactZero (none)

def event43362 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37501⟩⟩) ⟨⟨102⟩, ⟨84⟩, ⟨135⟩⟩ ⟨43204, 43362⟩

def event43363 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38355⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (1) 0 2 (.universal 43362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (none) 43361)

def event43364 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38355⟩⟩, .relation 43363 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩)

def event43365 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38355⟩⟩, .relation 43363 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩)

def event43366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38355⟩⟩, .relation 43363 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩)

def event43367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38355⟩⟩, .relation 43363 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact43368RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43368RawTermsValid :
    exact43368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38355⟩⟩) exact43368RawTerms .large 43200 (.finite 202072841853861888) (some (43202))

def event43369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39531⟩⟩) 0 ⟨38355⟩ 43368

def event43370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39531⟩⟩) 1 ⟨39530⟩ 43190

def event43371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39531⟩⟩) (.sum [.predecessor 0 43369 .coefficient, .predecessor 1 43370 .coefficient])

def event43372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39531⟩⟩, .operator (⟨43368, 0⟩, ⟨43190, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩, (1)⟩)

def event43373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39531⟩⟩, .operator (⟨43368, 2⟩, ⟨43190, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩, (-1)⟩)

def event43374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39531⟩⟩) (.sum [.result 43368 .summary, .result 43190 .summary])

def exact43375RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact43375RawTermsValid :
    exact43375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39531⟩⟩) exact43375RawTerms .large 43371 (.finite 32192736221397454434328420548608) (some (43374))

def event43376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39532⟩⟩) 0 ⟨39531⟩ 43375

def event43377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39532⟩⟩) 1 ⟨7162⟩ 15622

def event43378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39532⟩⟩) (.product (.predecessor 0 43376 .coefficient) (.predecessor 1 43377 .coefficient) (⟨false, false, none, none, none⟩))

def event43379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39532⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) [⟨.result 15618 .coefficient, false, none⟩])

def event43380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39532⟩⟩) (.product (.result 43375 .summary) (.transfer 43379) (⟨false, false, none, none, none⟩))

def event43381 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39532⟩⟩, .operator (⟨43375, 0⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event43382 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39532⟩⟩, .operator (⟨43375, 1⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event43383 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39532⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615)

def event43384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39532⟩⟩, .relation 43383 0, ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact43385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩]

theorem exact43385RawTermsValid :
    exact43385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39532⟩⟩) exact43385RawTerms .large 43378 (.finite 345666873099141705532726864949014345809920) (some (43380))

def event43386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35981⟩⟩) 0 ⟨7177⟩ 15500

def event43387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35981⟩⟩) 1 ⟨35980⟩ 34432

def event43388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35981⟩⟩) (.authority (.operator))

def exact43389RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35981⟩⟩]⟩, (1)⟩]

theorem exact43389RawTermsValid :
    exact43389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43389 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35981⟩⟩) exact43389RawTerms .large 43388 .exactZero (none)

def event43390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36848⟩⟩) 0 ⟨35981⟩ 43389

def event43391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36848⟩⟩) (.authority (.operator))

def exact43392RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩, (1)⟩]

theorem exact43392RawTermsValid :
    exact43392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36848⟩⟩) exact43392RawTerms (.finite 8192) 43391 .exactZero (none)

def event43393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36850⟩⟩) 0 ⟨36360⟩ 34716

def event43394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36850⟩⟩) 1 ⟨36848⟩ 43392

def event43395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36850⟩⟩) (.product (.predecessor 0 43393 .coefficient) (.predecessor 1 43394 .coefficient) (⟨false, false, none, none, none⟩))

def event43396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36850⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩) [⟨.result 43392 .coefficient, false, none⟩])

def event43397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36850⟩⟩) (.product (.result 34716 .summary) (.transfer 43396) (⟨false, false, none, none, none⟩))

def event43398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36850⟩⟩, .operator (⟨34716, 0⟩, ⟨43392, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩, (1)⟩)

def event43399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36850⟩⟩, .operator (⟨34716, 1⟩, ⟨43392, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩, (-1)⟩)

def event43400 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36850⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36848⟩⟩) ⟨35981⟩ 43389)

def event43401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36850⟩⟩, .relation 43400 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨35981⟩⟩]⟩, (-1)⟩)

def exact43402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨35981⟩⟩]⟩, (-1)⟩]

theorem exact43402RawTermsValid :
    exact43402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36850⟩⟩) exact43402RawTerms .large 43395 (.finite 32192539770951564984245676933120) (some (43397))

def event43403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35672⟩⟩) 0 ⟨34821⟩ 974

def event43404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35672⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact43405RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩]

theorem exact43405RawTermsValid :
    exact43405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35672⟩⟩) exact43405RawTerms (.finite 5647228698) 43404 .exactZero (none)

def event43406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35674⟩⟩) 0 ⟨35672⟩ 43405

def event43407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35674⟩⟩) 1 ⟨2370⟩ 4

def event43408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35674⟩⟩) (.scale (.predecessor 0 43406 .coefficient) (.value (.predecessor 1 43407 .coefficient)))

def exact43409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩]

theorem exact43409RawTermsValid :
    exact43409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43409 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35674⟩⟩) exact43409RawTerms (.finite 5647228698) 43408 .exactZero (none)

def event43410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35675⟩⟩) 0 ⟨11643⟩ 32120

def event43411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35675⟩⟩) 1 ⟨35674⟩ 43409

def event43412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35675⟩⟩) (.product (.predecessor 0 43410 .coefficient) (.predecessor 1 43411 .coefficient) (⟨false, false, none, none, none⟩))

def event43413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35675⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩) [⟨.result 43405 .coefficient, false, none⟩])

def event43414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35675⟩⟩) (.product (.result 32120 .summary) (.transfer 43413) (⟨false, false, none, none, none⟩))

def event43415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35675⟩⟩, .operator (⟨32120, 0⟩, ⟨43409, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩)

def event43416 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35673⟩⟩)

def event43417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event43418 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event43419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event43420 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event43421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event43422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event43423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43424 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43424

def event43426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 43422

def event43427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43425 .coefficient) (.value (.predecessor 1 43426 .coefficient)))

def event43428 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43428

def event43430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 43420

def event43431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43429 .coefficient, .predecessor 1 43430 .coefficient])

def event43432 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43432

def event43434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 43418

def event43435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43434 .coefficient))

def event43436 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34650⟩⟩) 0 ⟨11600⟩ 43436

def event43438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34650⟩⟩) (.authority (.programFamilyFact))

def exact43439RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩, (1)⟩]

theorem exact43439RawTermsValid :
    exact43439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34650⟩⟩) exact43439RawTerms (.finite 40) 43438 .exactZero (none)

def event43440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13716⟩⟩) 0 ⟨11600⟩ 43436

def event43441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13716⟩⟩) (.authority (.programFamilyFact))

def exact43442RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩], []⟩, (1)⟩]

theorem exact43442RawTermsValid :
    exact43442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13716⟩⟩) exact43442RawTerms (.finite 40) 43441 .exactZero (none)

def event43443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34651⟩⟩) 0 ⟨13716⟩ 43442

def event43444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34651⟩⟩) 1 ⟨34650⟩ 43439

def event43445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34651⟩⟩) (.product (.predecessor 0 43443 .coefficient) (.predecessor 1 43444 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34651⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩) [⟨.result 43442 .coefficient, true, some 1⟩, ⟨.result 43439 .coefficient, true, some 1⟩])

def event43447 : Event := .survivorFold (1) 43446

def exact43448RawTerms : List Term := []

theorem exact43448RawTermsValid :
    exact43448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43448 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34651⟩⟩) exact43448RawTerms (.finite 1600) 43445 (.finite 1600) (some (43446))

def event43449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34652⟩⟩) 0 ⟨34651⟩ 43448

def event43450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34652⟩⟩) (.identity (.predecessor 0 43449 .coefficient))

def event43451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34652⟩⟩) (.finite 1600)

def event43452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34820⟩⟩) 0 ⟨34652⟩ 43451

def event43453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34820⟩⟩) (.authority (.programFamilyFact))

def exact43454RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], []⟩, (1)⟩]

theorem exact43454RawTermsValid :
    exact43454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34820⟩⟩) exact43454RawTerms (.finite 40) 43453 .exactZero (none)

def event43455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34821⟩⟩) 0 ⟨34820⟩ 43454

def event43456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34821⟩⟩) (.identity (.predecessor 0 43455 .coefficient))

def event43457 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34821⟩⟩) (.finite 40)

def event43458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35672⟩⟩) 0 ⟨34821⟩ 43457

def event43459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35672⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact43460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩]

theorem exact43460RawTermsValid :
    exact43460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35672⟩⟩) exact43460RawTerms (.finite 5647228698) 43459 .exactZero (none)

def event43461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact43462RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact43462RawTermsValid :
    exact43462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact43462RawTerms .large 43461 .exactZero (none)

def event43463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35673⟩⟩) 0 ⟨35⟩ 43462

def event43464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35673⟩⟩) 1 ⟨35672⟩ 43460

def event43465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35673⟩⟩) (.product (.predecessor 0 43463 .coefficient) (.predecessor 1 43464 .coefficient) (⟨false, false, none, none, none⟩))

def event43466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35673⟩⟩, .operator (⟨43462, 0⟩, ⟨43460, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩)

def exact43467RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩]

theorem exact43467RawTermsValid :
    exact43467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43467 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35673⟩⟩) exact43467RawTerms .large 43465 .exactZero (none)

def event43468 : Event := .preFoldPolynomial 43467 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩] .exactZero none

def exact43469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35672⟩⟩]⟩, (1)⟩]

def event43469 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35673⟩⟩) 43468 exact43469RawTerms .large 43465 .exactZero (none)

def event43470 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36853⟩⟩)

def event43471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event43472 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event43473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event43474 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event43475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event43476 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event43477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43478 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43478

def event43480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 43476

def event43481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43479 .coefficient) (.value (.predecessor 1 43480 .coefficient)))

def event43482 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43482

def event43484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 43474

def event43485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43483 .coefficient, .predecessor 1 43484 .coefficient])

def event43486 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event43487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 43486

def event43488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 43472

def event43489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 43488 .coefficient))

def event43490 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event43491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34650⟩⟩) 0 ⟨11600⟩ 43490

def event43492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34650⟩⟩) (.authority (.programFamilyFact))

def exact43493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩, (1)⟩]

theorem exact43493RawTermsValid :
    exact43493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34650⟩⟩) exact43493RawTerms (.finite 40) 43492 .exactZero (none)

def event43494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13716⟩⟩) 0 ⟨11600⟩ 43490

def event43495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13716⟩⟩) (.authority (.programFamilyFact))

def exact43496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩], []⟩, (1)⟩]

theorem exact43496RawTermsValid :
    exact43496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13716⟩⟩) exact43496RawTerms (.finite 40) 43495 .exactZero (none)

def event43497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34651⟩⟩) 0 ⟨13716⟩ 43496

def event43498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34651⟩⟩) 1 ⟨34650⟩ 43493

def event43499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34651⟩⟩) (.product (.predecessor 0 43497 .coefficient) (.predecessor 1 43498 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event43500 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34651⟩⟩, .operator (⟨43496, 0⟩, ⟨43493, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩, (1)⟩)

def exact43501RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩, (1)⟩]

theorem exact43501RawTermsValid :
    exact43501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43501 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34651⟩⟩) exact43501RawTerms (.finite 1600) 43499 .exactZero (none)

def event43502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34652⟩⟩) 0 ⟨34651⟩ 43501

def event43503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34652⟩⟩) (.identity (.predecessor 0 43502 .coefficient))

def event43504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34652⟩⟩) (.finite 1600)

def event43505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34820⟩⟩) 0 ⟨34652⟩ 43504

def event43506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34820⟩⟩) (.authority (.programFamilyFact))

def exact43507RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], []⟩, (1)⟩]

theorem exact43507RawTermsValid :
    exact43507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34820⟩⟩) exact43507RawTerms (.finite 40) 43506 .exactZero (none)

def event43508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34821⟩⟩) 0 ⟨34820⟩ 43507

def event43509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34821⟩⟩) (.identity (.predecessor 0 43508 .coefficient))

def event43510 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34821⟩⟩) (.finite 40)

def event43511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35980⟩⟩) 0 ⟨34821⟩ 43510

def event43512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35980⟩⟩) (.authority (.programFamilyFact))

def event43513 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35980⟩⟩) (.finite 3720)

def event43514 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event43515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35981⟩⟩) 0 ⟨7177⟩ 43514

def event43516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35981⟩⟩) 1 ⟨35980⟩ 43513

def event43517 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35981⟩⟩) (.authority (.operator))

def exact43518RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35981⟩⟩]⟩, (1)⟩]

theorem exact43518RawTermsValid :
    exact43518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event43518 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35981⟩⟩) exact43518RawTerms .large 43517 .exactZero (none)

def event43519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36848⟩⟩) 0 ⟨35981⟩ 43518

def eventLeaf2704 : Array AnnotatedEvent := #[
  { event := event43264
    frameStart := 43258 },
  { event := event43265
    frameStart := 43258 },
  { event := event43266
    frameStart := 43258 },
  { event := event43267
    frameStart := 43258 },
  { event := event43268
    frameStart := 43258 },
  { event := event43269
    frameStart := 43258 },
  { event := event43270
    frameStart := 43258 },
  { event := event43271
    frameStart := 43258 },
  { event := event43272
    frameStart := 43258 },
  { event := event43273
    frameStart := 43258 },
  { event := event43274
    frameStart := 43258 },
  { event := event43275
    frameStart := 43258 },
  { event := event43276
    frameStart := 43258 },
  { event := event43277
    frameStart := 43258 },
  { event := event43278
    frameStart := 43258 },
  { event := event43279
    frameStart := 43258 }
]

def eventLeaf2705 : Array AnnotatedEvent := #[
  { event := event43280
    frameStart := 43258 },
  { event := event43281
    frameStart := 43258 },
  { event := event43282
    frameStart := 43258 },
  { event := event43283
    frameStart := 43258 },
  { event := event43284
    frameStart := 43258 },
  { event := event43285
    frameStart := 43258 },
  { event := event43286
    frameStart := 43258 },
  { event := event43287
    frameStart := 43258 },
  { event := event43288
    frameStart := 43258 },
  { event := event43289
    frameStart := 43258 },
  { event := event43290
    frameStart := 43258 },
  { event := event43291
    frameStart := 43258 },
  { event := event43292
    frameStart := 43258 },
  { event := event43293
    frameStart := 43258 },
  { event := event43294
    frameStart := 43258 },
  { event := event43295
    frameStart := 43258 }
]

def eventLeaf2706 : Array AnnotatedEvent := #[
  { event := event43296
    frameStart := 43258 },
  { event := event43297
    frameStart := 43258 },
  { event := event43298
    frameStart := 43258 },
  { event := event43299
    frameStart := 43258 },
  { event := event43300
    frameStart := 43258 },
  { event := event43301
    frameStart := 43258 },
  { event := event43302
    frameStart := 43258 },
  { event := event43303
    frameStart := 43258 },
  { event := event43304
    frameStart := 43258 },
  { event := event43305
    frameStart := 43258 },
  { event := event43306
    frameStart := 43258 },
  { event := event43307
    frameStart := 43258 },
  { event := event43308
    frameStart := 43258 },
  { event := event43309
    frameStart := 43258 },
  { event := event43310
    frameStart := 43258 },
  { event := event43311
    frameStart := 43258 }
]

def eventLeaf2707 : Array AnnotatedEvent := #[
  { event := event43312
    frameStart := 43258 },
  { event := event43313
    frameStart := 43258 },
  { event := event43314
    frameStart := 43258 },
  { event := event43315
    frameStart := 43258 },
  { event := event43316
    frameStart := 43258 },
  { event := event43317
    frameStart := 43258 },
  { event := event43318
    frameStart := 43258 },
  { event := event43319
    frameStart := 43258 },
  { event := event43320
    frameStart := 43258 },
  { event := event43321
    frameStart := 43258 },
  { event := event43322
    frameStart := 43258 },
  { event := event43323
    frameStart := 43258 },
  { event := event43324
    frameStart := 43258 },
  { event := event43325
    frameStart := 43258 },
  { event := event43326
    frameStart := 43258 },
  { event := event43327
    frameStart := 43258 }
]

def eventLeaf2708 : Array AnnotatedEvent := #[
  { event := event43328
    frameStart := 43258 },
  { event := event43329
    frameStart := 43258 },
  { event := event43330
    frameStart := 43258 },
  { event := event43331
    frameStart := 43258 },
  { event := event43332
    frameStart := 43258 },
  { event := event43333
    frameStart := 43258 },
  { event := event43334
    frameStart := 43258 },
  { event := event43335
    frameStart := 43258 },
  { event := event43336
    frameStart := 43258 },
  { event := event43337
    frameStart := 43258 },
  { event := event43338
    frameStart := 43258 },
  { event := event43339
    frameStart := 43258 },
  { event := event43340
    frameStart := 43258 },
  { event := event43341
    frameStart := 43258 },
  { event := event43342
    frameStart := 43258 },
  { event := event43343
    frameStart := 43258 }
]

def eventLeaf2709 : Array AnnotatedEvent := #[
  { event := event43344
    frameStart := 43258 },
  { event := event43345
    frameStart := 43258 },
  { event := event43346
    frameStart := 43258 },
  { event := event43347
    frameStart := 43258 },
  { event := event43348
    frameStart := 43258 },
  { event := event43349
    frameStart := 43258 },
  { event := event43350
    frameStart := 43258 },
  { event := event43351
    frameStart := 43258 },
  { event := event43352
    frameStart := 43258 },
  { event := event43353
    frameStart := 43258 },
  { event := event43354
    frameStart := 43258 },
  { event := event43355
    frameStart := 43258 },
  { event := event43356
    frameStart := 43258 },
  { event := event43357
    frameStart := 43258 },
  { event := event43358
    frameStart := 43258 },
  { event := event43359
    frameStart := 43258 }
]

def eventLeaf2710 : Array AnnotatedEvent := #[
  { event := event43360
    frameStart := 43258 },
  { event := event43361
    frameStart := 43258 },
  { event := event43362
    frameStart := 0 },
  { event := event43363
    frameStart := 0 },
  { event := event43364
    frameStart := 0 },
  { event := event43365
    frameStart := 0 },
  { event := event43366
    frameStart := 0 },
  { event := event43367
    frameStart := 0 },
  { event := event43368
    frameStart := 0 },
  { event := event43369
    frameStart := 0 },
  { event := event43370
    frameStart := 0 },
  { event := event43371
    frameStart := 0 },
  { event := event43372
    frameStart := 0 },
  { event := event43373
    frameStart := 0 },
  { event := event43374
    frameStart := 0 },
  { event := event43375
    frameStart := 0 }
]

def eventLeaf2711 : Array AnnotatedEvent := #[
  { event := event43376
    frameStart := 0 },
  { event := event43377
    frameStart := 0 },
  { event := event43378
    frameStart := 0 },
  { event := event43379
    frameStart := 0 },
  { event := event43380
    frameStart := 0 },
  { event := event43381
    frameStart := 0 },
  { event := event43382
    frameStart := 0 },
  { event := event43383
    frameStart := 0 },
  { event := event43384
    frameStart := 0 },
  { event := event43385
    frameStart := 0 },
  { event := event43386
    frameStart := 0 },
  { event := event43387
    frameStart := 0 },
  { event := event43388
    frameStart := 0 },
  { event := event43389
    frameStart := 0 },
  { event := event43390
    frameStart := 0 },
  { event := event43391
    frameStart := 0 }
]

def eventLeaf2712 : Array AnnotatedEvent := #[
  { event := event43392
    frameStart := 0 },
  { event := event43393
    frameStart := 0 },
  { event := event43394
    frameStart := 0 },
  { event := event43395
    frameStart := 0 },
  { event := event43396
    frameStart := 0 },
  { event := event43397
    frameStart := 0 },
  { event := event43398
    frameStart := 0 },
  { event := event43399
    frameStart := 0 },
  { event := event43400
    frameStart := 0 },
  { event := event43401
    frameStart := 0 },
  { event := event43402
    frameStart := 0 },
  { event := event43403
    frameStart := 0 },
  { event := event43404
    frameStart := 0 },
  { event := event43405
    frameStart := 0 },
  { event := event43406
    frameStart := 0 },
  { event := event43407
    frameStart := 0 }
]

def eventLeaf2713 : Array AnnotatedEvent := #[
  { event := event43408
    frameStart := 0 },
  { event := event43409
    frameStart := 0 },
  { event := event43410
    frameStart := 0 },
  { event := event43411
    frameStart := 0 },
  { event := event43412
    frameStart := 0 },
  { event := event43413
    frameStart := 0 },
  { event := event43414
    frameStart := 0 },
  { event := event43415
    frameStart := 0 },
  { event := event43416
    frameStart := 43416 },
  { event := event43417
    frameStart := 43416 },
  { event := event43418
    frameStart := 43416 },
  { event := event43419
    frameStart := 43416 },
  { event := event43420
    frameStart := 43416 },
  { event := event43421
    frameStart := 43416 },
  { event := event43422
    frameStart := 43416 },
  { event := event43423
    frameStart := 43416 }
]

def eventLeaf2714 : Array AnnotatedEvent := #[
  { event := event43424
    frameStart := 43416 },
  { event := event43425
    frameStart := 43416 },
  { event := event43426
    frameStart := 43416 },
  { event := event43427
    frameStart := 43416 },
  { event := event43428
    frameStart := 43416 },
  { event := event43429
    frameStart := 43416 },
  { event := event43430
    frameStart := 43416 },
  { event := event43431
    frameStart := 43416 },
  { event := event43432
    frameStart := 43416 },
  { event := event43433
    frameStart := 43416 },
  { event := event43434
    frameStart := 43416 },
  { event := event43435
    frameStart := 43416 },
  { event := event43436
    frameStart := 43416 },
  { event := event43437
    frameStart := 43416 },
  { event := event43438
    frameStart := 43416 },
  { event := event43439
    frameStart := 43416 }
]

def eventLeaf2715 : Array AnnotatedEvent := #[
  { event := event43440
    frameStart := 43416 },
  { event := event43441
    frameStart := 43416 },
  { event := event43442
    frameStart := 43416 },
  { event := event43443
    frameStart := 43416 },
  { event := event43444
    frameStart := 43416 },
  { event := event43445
    frameStart := 43416 },
  { event := event43446
    frameStart := 43416 },
  { event := event43447
    frameStart := 43416 },
  { event := event43448
    frameStart := 43416 },
  { event := event43449
    frameStart := 43416 },
  { event := event43450
    frameStart := 43416 },
  { event := event43451
    frameStart := 43416 },
  { event := event43452
    frameStart := 43416 },
  { event := event43453
    frameStart := 43416 },
  { event := event43454
    frameStart := 43416 },
  { event := event43455
    frameStart := 43416 }
]

def eventLeaf2716 : Array AnnotatedEvent := #[
  { event := event43456
    frameStart := 43416 },
  { event := event43457
    frameStart := 43416 },
  { event := event43458
    frameStart := 43416 },
  { event := event43459
    frameStart := 43416 },
  { event := event43460
    frameStart := 43416 },
  { event := event43461
    frameStart := 43416 },
  { event := event43462
    frameStart := 43416 },
  { event := event43463
    frameStart := 43416 },
  { event := event43464
    frameStart := 43416 },
  { event := event43465
    frameStart := 43416 },
  { event := event43466
    frameStart := 43416 },
  { event := event43467
    frameStart := 43416 },
  { event := event43468
    frameStart := 43416 },
  { event := event43469
    frameStart := 43416 },
  { event := event43470
    frameStart := 43470 },
  { event := event43471
    frameStart := 43470 }
]

def eventLeaf2717 : Array AnnotatedEvent := #[
  { event := event43472
    frameStart := 43470 },
  { event := event43473
    frameStart := 43470 },
  { event := event43474
    frameStart := 43470 },
  { event := event43475
    frameStart := 43470 },
  { event := event43476
    frameStart := 43470 },
  { event := event43477
    frameStart := 43470 },
  { event := event43478
    frameStart := 43470 },
  { event := event43479
    frameStart := 43470 },
  { event := event43480
    frameStart := 43470 },
  { event := event43481
    frameStart := 43470 },
  { event := event43482
    frameStart := 43470 },
  { event := event43483
    frameStart := 43470 },
  { event := event43484
    frameStart := 43470 },
  { event := event43485
    frameStart := 43470 },
  { event := event43486
    frameStart := 43470 },
  { event := event43487
    frameStart := 43470 }
]

def eventLeaf2718 : Array AnnotatedEvent := #[
  { event := event43488
    frameStart := 43470 },
  { event := event43489
    frameStart := 43470 },
  { event := event43490
    frameStart := 43470 },
  { event := event43491
    frameStart := 43470 },
  { event := event43492
    frameStart := 43470 },
  { event := event43493
    frameStart := 43470 },
  { event := event43494
    frameStart := 43470 },
  { event := event43495
    frameStart := 43470 },
  { event := event43496
    frameStart := 43470 },
  { event := event43497
    frameStart := 43470 },
  { event := event43498
    frameStart := 43470 },
  { event := event43499
    frameStart := 43470 },
  { event := event43500
    frameStart := 43470 },
  { event := event43501
    frameStart := 43470 },
  { event := event43502
    frameStart := 43470 },
  { event := event43503
    frameStart := 43470 }
]

def eventLeaf2719 : Array AnnotatedEvent := #[
  { event := event43504
    frameStart := 43470 },
  { event := event43505
    frameStart := 43470 },
  { event := event43506
    frameStart := 43470 },
  { event := event43507
    frameStart := 43470 },
  { event := event43508
    frameStart := 43470 },
  { event := event43509
    frameStart := 43470 },
  { event := event43510
    frameStart := 43470 },
  { event := event43511
    frameStart := 43470 },
  { event := event43512
    frameStart := 43470 },
  { event := event43513
    frameStart := 43470 },
  { event := event43514
    frameStart := 43470 },
  { event := event43515
    frameStart := 43470 },
  { event := event43516
    frameStart := 43470 },
  { event := event43517
    frameStart := 43470 },
  { event := event43518
    frameStart := 43470 },
  { event := event43519
    frameStart := 43470 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events169
