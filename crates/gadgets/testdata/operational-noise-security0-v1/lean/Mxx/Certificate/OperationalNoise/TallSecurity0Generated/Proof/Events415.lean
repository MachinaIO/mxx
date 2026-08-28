import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events415

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact106240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (-1)⟩]

theorem exact106240RawTermsValid :
    exact106240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27175⟩⟩) exact106240RawTerms .large 106233 (.finite 1291978822348200476672) (some (106235))

def event106241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20885⟩⟩) 0 ⟨15574⟩ 4882

def event106242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20885⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact106243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩]

theorem exact106243RawTermsValid :
    exact106243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20885⟩⟩) exact106243RawTerms (.finite 136065468) 106242 .exactZero (none)

def event106244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20887⟩⟩) 0 ⟨20885⟩ 106243

def event106245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20887⟩⟩) 1 ⟨2348⟩ 4

def event106246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20887⟩⟩) (.scale (.predecessor 0 106244 .coefficient) (.value (.predecessor 1 106245 .coefficient)))

def exact106247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩]

theorem exact106247RawTermsValid :
    exact106247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20887⟩⟩) exact106247RawTerms (.finite 136065468) 106246 .exactZero (none)

def event106248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20888⟩⟩) 0 ⟨5509⟩ 94462

def event106249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20888⟩⟩) 1 ⟨20887⟩ 106247

def event106250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20888⟩⟩) (.product (.predecessor 0 106248 .coefficient) (.predecessor 1 106249 .coefficient) (⟨false, false, none, none, none⟩))

def event106251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20888⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩) [⟨.result 106243 .coefficient, false, none⟩])

def event106252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20888⟩⟩) (.product (.result 94462 .summary) (.transfer 106251) (⟨false, false, none, none, none⟩))

def event106253 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20888⟩⟩, .operator (⟨94462, 0⟩, ⟨106247, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩)

def event106254 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20886⟩⟩)

def event106255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106256 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106258 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106258

def event106260 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106256

def event106261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106259 .coefficient) (.value (.predecessor 1 106260 .coefficient)))

def event106262 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11205⟩⟩) 0 ⟨5503⟩ 106262

def event106264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11205⟩⟩) (.authority (.programFamilyFact))

def exact106265RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩], []⟩, (1)⟩]

theorem exact106265RawTermsValid :
    exact106265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106265 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11205⟩⟩) exact106265RawTerms (.finite 10) 106264 .exactZero (none)

def event106266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13529⟩⟩) 0 ⟨5503⟩ 106262

def event106267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13529⟩⟩) (.authority (.programFamilyFact))

def exact106268RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩, (1)⟩]

theorem exact106268RawTermsValid :
    exact106268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106268 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13529⟩⟩) exact106268RawTerms (.finite 10) 106267 .exactZero (none)

def event106269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13530⟩⟩) 0 ⟨13529⟩ 106268

def event106270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13530⟩⟩) 1 ⟨11205⟩ 106265

def event106271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13530⟩⟩) (.product (.predecessor 0 106269 .coefficient) (.predecessor 1 106270 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13530⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩) [⟨.result 106268 .coefficient, true, some 1⟩, ⟨.result 106265 .coefficient, true, some 1⟩])

def event106273 : Event := .survivorFold (1) 106272

def exact106274RawTerms : List Term := []

theorem exact106274RawTermsValid :
    exact106274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106274 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13530⟩⟩) exact106274RawTerms (.finite 100) 106271 (.finite 100) (some (106272))

def event106275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13531⟩⟩) 0 ⟨13530⟩ 106274

def event106276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13531⟩⟩) (.identity (.predecessor 0 106275 .coefficient))

def event106277 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13531⟩⟩) (.finite 100)

def event106278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15573⟩⟩) 0 ⟨13531⟩ 106277

def event106279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15573⟩⟩) (.authority (.programFamilyFact))

def exact106280RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], []⟩, (1)⟩]

theorem exact106280RawTermsValid :
    exact106280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106280 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15573⟩⟩) exact106280RawTerms (.finite 10) 106279 .exactZero (none)

def event106281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15574⟩⟩) 0 ⟨15573⟩ 106280

def event106282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15574⟩⟩) (.identity (.predecessor 0 106281 .coefficient))

def event106283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15574⟩⟩) (.finite 10)

def event106284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20885⟩⟩) 0 ⟨15574⟩ 106283

def event106285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20885⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact106286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩]

theorem exact106286RawTermsValid :
    exact106286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106286 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20885⟩⟩) exact106286RawTerms (.finite 136065468) 106285 .exactZero (none)

def event106287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact106288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact106288RawTermsValid :
    exact106288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact106288RawTerms .large 106287 .exactZero (none)

def event106289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20886⟩⟩) 0 ⟨6⟩ 106288

def event106290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20886⟩⟩) 1 ⟨20885⟩ 106286

def event106291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20886⟩⟩) (.product (.predecessor 0 106289 .coefficient) (.predecessor 1 106290 .coefficient) (⟨false, false, none, none, none⟩))

def event106292 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20886⟩⟩, .operator (⟨106288, 0⟩, ⟨106286, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩)

def exact106293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩]

theorem exact106293RawTermsValid :
    exact106293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106293 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20886⟩⟩) exact106293RawTerms .large 106291 .exactZero (none)

def event106294 : Event := .preFoldPolynomial 106293 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩] .exactZero none

def exact106295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩, (1)⟩]

def event106295 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20886⟩⟩) 106294 exact106295RawTerms .large 106291 .exactZero (none)

def event106296 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27179⟩⟩)

def event106297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106298 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106300 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106300

def event106302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106298

def event106303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106301 .coefficient) (.value (.predecessor 1 106302 .coefficient)))

def event106304 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11205⟩⟩) 0 ⟨5503⟩ 106304

def event106306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11205⟩⟩) (.authority (.programFamilyFact))

def exact106307RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩], []⟩, (1)⟩]

theorem exact106307RawTermsValid :
    exact106307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11205⟩⟩) exact106307RawTerms (.finite 10) 106306 .exactZero (none)

def event106308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13529⟩⟩) 0 ⟨5503⟩ 106304

def event106309 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13529⟩⟩) (.authority (.programFamilyFact))

def exact106310RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩, (1)⟩]

theorem exact106310RawTermsValid :
    exact106310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106310 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13529⟩⟩) exact106310RawTerms (.finite 10) 106309 .exactZero (none)

def event106311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13530⟩⟩) 0 ⟨13529⟩ 106310

def event106312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13530⟩⟩) 1 ⟨11205⟩ 106307

def event106313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13530⟩⟩) (.product (.predecessor 0 106311 .coefficient) (.predecessor 1 106312 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106314 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13530⟩⟩, .operator (⟨106310, 0⟩, ⟨106307, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩, (1)⟩)

def exact106315RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩, (1)⟩]

theorem exact106315RawTermsValid :
    exact106315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106315 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13530⟩⟩) exact106315RawTerms (.finite 100) 106313 .exactZero (none)

def event106316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13531⟩⟩) 0 ⟨13530⟩ 106315

def event106317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13531⟩⟩) (.identity (.predecessor 0 106316 .coefficient))

def event106318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13531⟩⟩) (.finite 100)

def event106319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15573⟩⟩) 0 ⟨13531⟩ 106318

def event106320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15573⟩⟩) (.authority (.programFamilyFact))

def exact106321RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], []⟩, (1)⟩]

theorem exact106321RawTermsValid :
    exact106321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15573⟩⟩) exact106321RawTerms (.finite 10) 106320 .exactZero (none)

def event106322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15574⟩⟩) 0 ⟨15573⟩ 106321

def event106323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15574⟩⟩) (.identity (.predecessor 0 106322 .coefficient))

def event106324 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15574⟩⟩) (.finite 10)

def event106325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23962⟩⟩) 0 ⟨15574⟩ 106324

def event106326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23962⟩⟩) (.authority (.programFamilyFact))

def event106327 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23962⟩⟩) (.finite 3720)

def event106328 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event106329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23963⟩⟩) 0 ⟨6689⟩ 106328

def event106330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23963⟩⟩) 1 ⟨23962⟩ 106327

def event106331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23963⟩⟩) (.authority (.operator))

def exact106332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩]

theorem exact106332RawTermsValid :
    exact106332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23963⟩⟩) exact106332RawTerms .large 106331 .exactZero (none)

def event106333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27173⟩⟩) 0 ⟨23963⟩ 106332

def event106334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27173⟩⟩) (.authority (.operator))

def exact106335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩]

theorem exact106335RawTermsValid :
    exact106335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106335 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27173⟩⟩) exact106335RawTerms (.finite 8192) 106334 .exactZero (none)

def event106336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event106337 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event106338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15650⟩⟩) 0 ⟨15574⟩ 106324

def event106339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15650⟩⟩) 1 ⟨110⟩ 106337

def event106340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15650⟩⟩) (.sum [.predecessor 0 106338 .coefficient, .predecessor 1 106339 .coefficient])

def event106341 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15650⟩⟩) (.finite 10)

def event106342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15651⟩⟩) 0 ⟨15650⟩ 106341

def event106343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15651⟩⟩) (.identity (.predecessor 0 106342 .coefficient))

def exact106344RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], []⟩, (1)⟩]

theorem exact106344RawTermsValid :
    exact106344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15651⟩⟩) exact106344RawTerms (.finite 10) 106343 .exactZero (none)

def event106345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact106346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106346RawTermsValid :
    exact106346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106346 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact106346RawTerms .large 106345 .exactZero (none)

def event106347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15652⟩⟩) 0 ⟨6544⟩ 106346

def event106348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15652⟩⟩) 1 ⟨15651⟩ 106344

def event106349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15652⟩⟩) (.product (.predecessor 0 106347 .coefficient) (.predecessor 1 106348 .coefficient) (⟨false, false, none, none, none⟩))

def event106350 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15652⟩⟩, .operator (⟨106346, 0⟩, ⟨106344, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106351RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106351RawTermsValid :
    exact106351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15652⟩⟩) exact106351RawTerms .large 106349 .exactZero (none)

def event106352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 106328

def event106353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact106354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact106354RawTermsValid :
    exact106354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106354 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact106354RawTerms .large 106353 .exactZero (none)

def event106355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15653⟩⟩) 0 ⟨6694⟩ 106354

def event106356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15653⟩⟩) 1 ⟨15652⟩ 106351

def event106357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15653⟩⟩) (.sum [.predecessor 0 106355 .coefficient, .predecessor 1 106356 .coefficient])

def exact106358RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106358RawTermsValid :
    exact106358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106358 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15653⟩⟩) exact106358RawTerms .large 106357 .exactZero (none)

def event106359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27174⟩⟩) 0 ⟨15653⟩ 106358

def event106360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27174⟩⟩) 1 ⟨27173⟩ 106335

def event106361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27174⟩⟩) (.product (.predecessor 0 106359 .coefficient) (.predecessor 1 106360 .coefficient) (⟨false, false, none, none, none⟩))

def event106362 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27174⟩⟩, .operator (⟨106358, 0⟩, ⟨106335, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩)

def event106363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27174⟩⟩, .operator (⟨106358, 1⟩, ⟨106335, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩)

def event106364 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27174⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27173⟩⟩) ⟨23963⟩ 106332)

def event106365 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27174⟩⟩, .relation 106364 0, ⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (-1)⟩)

def exact106366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (-1)⟩]

theorem exact106366RawTermsValid :
    exact106366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106366 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27174⟩⟩) exact106366RawTerms .large 106361 .exactZero (none)

def event106367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17792⟩⟩) 0 ⟨15574⟩ 106324

def event106368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17792⟩⟩) (.authority (.programFamilyFact))

def exact106369RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], []⟩, (1)⟩]

theorem exact106369RawTermsValid :
    exact106369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17792⟩⟩) exact106369RawTerms (.finite 10) 106368 .exactZero (none)

def event106370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17798⟩⟩) 0 ⟨6544⟩ 106346

def event106371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17798⟩⟩) 1 ⟨17792⟩ 106369

def event106372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17798⟩⟩) (.product (.predecessor 0 106370 .coefficient) (.predecessor 1 106371 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106373 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17798⟩⟩, .operator (⟨106346, 0⟩, ⟨106369, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact106374RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact106374RawTermsValid :
    exact106374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17798⟩⟩) exact106374RawTerms .large 106372 .exactZero (none)

def event106375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6716⟩⟩) 0 ⟨6689⟩ 106328

def event106376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6716⟩⟩) (.authority (.operator))

def exact106377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩]

theorem exact106377RawTermsValid :
    exact106377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6716⟩⟩) exact106377RawTerms .large 106376 .exactZero (none)

def event106378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17799⟩⟩) 0 ⟨6716⟩ 106377

def event106379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17799⟩⟩) 1 ⟨17798⟩ 106374

def event106380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17799⟩⟩) (.sum [.predecessor 0 106378 .coefficient, .predecessor 1 106379 .coefficient])

def exact106381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106381RawTermsValid :
    exact106381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106381 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17799⟩⟩) exact106381RawTerms .large 106380 .exactZero (none)

def event106382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27179⟩⟩) 0 ⟨17799⟩ 106381

def event106383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27179⟩⟩) 1 ⟨27174⟩ 106366

def event106384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27179⟩⟩) (.sum [.predecessor 0 106382 .coefficient, .predecessor 1 106383 .coefficient])

def exact106385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106385RawTermsValid :
    exact106385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27179⟩⟩) exact106385RawTerms .large 106384 .exactZero (none)

def event106386 : Event := .preFoldPolynomial 106385 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event106387 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27179⟩⟩) 106386 exact106387RawTerms .large 106384 .exactZero (none)

def event106388 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15574⟩⟩) ⟨⟨129⟩, ⟨36⟩, ⟨109⟩⟩ ⟨106254, 106388⟩

def event106389 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20888⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩) (1) 0 2 (.universal 106388 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20885⟩⟩]⟩) (none) 106387)

def event106390 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20888⟩⟩, .relation 106389 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩)

def event106391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20888⟩⟩, .relation 106389 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩)

def event106392 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20888⟩⟩, .relation 106389 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩)

def event106393 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20888⟩⟩, .relation 106389 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106394RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106394RawTermsValid :
    exact106394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106394 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20888⟩⟩) exact106394RawTerms .large 106250 (.finite 1811303510016) (some (106252))

def event106395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27176⟩⟩) 0 ⟨20888⟩ 106394

def event106396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27176⟩⟩) 1 ⟨27175⟩ 106240

def event106397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27176⟩⟩) (.sum [.predecessor 0 106395 .coefficient, .predecessor 1 106396 .coefficient])

def event106398 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27176⟩⟩, .operator (⟨106394, 0⟩, ⟨106240, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27173⟩⟩]⟩, (1)⟩)

def event106399 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27176⟩⟩, .operator (⟨106394, 2⟩, ⟨106240, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15573⟩⟩], [⟨.program ⟨214⟩, ⟨23963⟩⟩]⟩, (-1)⟩)

def event106400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27176⟩⟩) (.sum [.result 106394 .summary, .result 106240 .summary])

def exact106401RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106401RawTermsValid :
    exact106401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106401 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27176⟩⟩) exact106401RawTerms .large 106397 (.finite 1291978824159503986688) (some (106400))

def event106402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27177⟩⟩) 0 ⟨27176⟩ 106401

def event106403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27177⟩⟩) 1 ⟨6650⟩ 5779

def event106404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27177⟩⟩) (.product (.predecessor 0 106402 .coefficient) (.predecessor 1 106403 .coefficient) (⟨false, false, none, none, none⟩))

def event106405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27177⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) [⟨.result 5775 .coefficient, false, none⟩])

def event106406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27177⟩⟩) (.product (.result 106401 .summary) (.transfer 106405) (⟨false, false, none, none, none⟩))

def event106407 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27177⟩⟩, .operator (⟨106401, 0⟩, ⟨5779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩)

def event106408 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27177⟩⟩, .operator (⟨106401, 1⟩, ⟨5779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (-1)⟩)

def event106409 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27177⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6649⟩⟩) ⟨6596⟩ 5772)

def event106410 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27177⟩⟩, .relation 106409 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact106411RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact106411RawTermsValid :
    exact106411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27177⟩⟩) exact106411RawTerms .large 106404 (.finite 4741582956326566183208747008) (some (106406))

def event106412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23900⟩⟩) 0 ⟨6689⟩ 5477

def event106413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23900⟩⟩) 1 ⟨23899⟩ 100440

def event106414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23900⟩⟩) (.authority (.operator))

def exact106415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23900⟩⟩]⟩, (1)⟩]

theorem exact106415RawTermsValid :
    exact106415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106415 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23900⟩⟩) exact106415RawTerms .large 106414 .exactZero (none)

def event106416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26956⟩⟩) 0 ⟨23900⟩ 106415

def event106417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26956⟩⟩) (.authority (.operator))

def exact106418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩, (1)⟩]

theorem exact106418RawTermsValid :
    exact106418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26956⟩⟩) exact106418RawTerms (.finite 8192) 106417 .exactZero (none)

def event106419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26958⟩⟩) 0 ⟨25285⟩ 100700

def event106420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26958⟩⟩) 1 ⟨26956⟩ 106418

def event106421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26958⟩⟩) (.product (.predecessor 0 106419 .coefficient) (.predecessor 1 106420 .coefficient) (⟨false, false, none, none, none⟩))

def event106422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26958⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩) [⟨.result 106418 .coefficient, false, none⟩])

def event106423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26958⟩⟩) (.product (.result 100700 .summary) (.transfer 106422) (⟨false, false, none, none, none⟩))

def event106424 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26958⟩⟩, .operator (⟨100700, 0⟩, ⟨106418, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩, (1)⟩)

def event106425 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26958⟩⟩, .operator (⟨100700, 1⟩, ⟨106418, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15412⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩, (-1)⟩)

def event106426 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26958⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15412⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26956⟩⟩) ⟨23900⟩ 106415)

def event106427 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26958⟩⟩, .relation 106426 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15412⟩⟩], [⟨.program ⟨214⟩, ⟨23900⟩⟩]⟩, (-1)⟩)

def exact106428RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26956⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15412⟩⟩], [⟨.program ⟨214⟩, ⟨23900⟩⟩]⟩, (-1)⟩]

theorem exact106428RawTermsValid :
    exact106428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106428 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26958⟩⟩) exact106428RawTerms .large 106421 (.finite 1291933997458159304704) (some (106423))

def event106429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20741⟩⟩) 0 ⟨15413⟩ 4905

def event106430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20741⟩⟩) (.authority (.relationPreimageSource ⟨34⟩))

def exact106431RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩]

theorem exact106431RawTermsValid :
    exact106431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20741⟩⟩) exact106431RawTerms (.finite 136065468) 106430 .exactZero (none)

def event106432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20743⟩⟩) 0 ⟨20741⟩ 106431

def event106433 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20743⟩⟩) 1 ⟨2348⟩ 4

def event106434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20743⟩⟩) (.scale (.predecessor 0 106432 .coefficient) (.value (.predecessor 1 106433 .coefficient)))

def exact106435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩]

theorem exact106435RawTermsValid :
    exact106435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106435 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20743⟩⟩) exact106435RawTerms (.finite 136065468) 106434 .exactZero (none)

def event106436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20744⟩⟩) 0 ⟨5509⟩ 94462

def event106437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20744⟩⟩) 1 ⟨20743⟩ 106435

def event106438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20744⟩⟩) (.product (.predecessor 0 106436 .coefficient) (.predecessor 1 106437 .coefficient) (⟨false, false, none, none, none⟩))

def event106439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20744⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩) [⟨.result 106431 .coefficient, false, none⟩])

def event106440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20744⟩⟩) (.product (.result 94462 .summary) (.transfer 106439) (⟨false, false, none, none, none⟩))

def event106441 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20744⟩⟩, .operator (⟨94462, 0⟩, ⟨106435, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩)

def event106442 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20742⟩⟩)

def event106443 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106444 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106446 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106446

def event106448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106444

def event106449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106447 .coefficient) (.value (.predecessor 1 106448 .coefficient)))

def event106450 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11121⟩⟩) 0 ⟨5503⟩ 106450

def event106452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11121⟩⟩) (.authority (.programFamilyFact))

def exact106453RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩], []⟩, (1)⟩]

theorem exact106453RawTermsValid :
    exact106453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11121⟩⟩) exact106453RawTerms (.finite 6) 106452 .exactZero (none)

def event106454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12136⟩⟩) 0 ⟨5503⟩ 106450

def event106455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12136⟩⟩) (.authority (.programFamilyFact))

def exact106456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12136⟩⟩], []⟩, (1)⟩]

theorem exact106456RawTermsValid :
    exact106456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12136⟩⟩) exact106456RawTerms (.finite 6) 106455 .exactZero (none)

def event106457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12137⟩⟩) 0 ⟨12136⟩ 106456

def event106458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12137⟩⟩) 1 ⟨11121⟩ 106453

def event106459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12137⟩⟩) (.product (.predecessor 0 106457 .coefficient) (.predecessor 1 106458 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12137⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], []⟩) [⟨.result 106456 .coefficient, true, some 1⟩, ⟨.result 106453 .coefficient, true, some 1⟩])

def event106461 : Event := .survivorFold (1) 106460

def exact106462RawTerms : List Term := []

theorem exact106462RawTermsValid :
    exact106462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12137⟩⟩) exact106462RawTerms (.finite 36) 106459 (.finite 36) (some (106460))

def event106463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12138⟩⟩) 0 ⟨12137⟩ 106462

def event106464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12138⟩⟩) (.identity (.predecessor 0 106463 .coefficient))

def event106465 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12138⟩⟩) (.finite 36)

def event106466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15412⟩⟩) 0 ⟨12138⟩ 106465

def event106467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15412⟩⟩) (.authority (.programFamilyFact))

def exact106468RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15412⟩⟩], []⟩, (1)⟩]

theorem exact106468RawTermsValid :
    exact106468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106468 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15412⟩⟩) exact106468RawTerms (.finite 6) 106467 .exactZero (none)

def event106469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15413⟩⟩) 0 ⟨15412⟩ 106468

def event106470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15413⟩⟩) (.identity (.predecessor 0 106469 .coefficient))

def event106471 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15413⟩⟩) (.finite 6)

def event106472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20741⟩⟩) 0 ⟨15413⟩ 106471

def event106473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20741⟩⟩) (.authority (.relationPreimageSource ⟨34⟩))

def exact106474RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩]

theorem exact106474RawTermsValid :
    exact106474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20741⟩⟩) exact106474RawTerms (.finite 136065468) 106473 .exactZero (none)

def event106475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact106476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact106476RawTermsValid :
    exact106476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact106476RawTerms .large 106475 .exactZero (none)

def event106477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20742⟩⟩) 0 ⟨6⟩ 106476

def event106478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20742⟩⟩) 1 ⟨20741⟩ 106474

def event106479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20742⟩⟩) (.product (.predecessor 0 106477 .coefficient) (.predecessor 1 106478 .coefficient) (⟨false, false, none, none, none⟩))

def event106480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20742⟩⟩, .operator (⟨106476, 0⟩, ⟨106474, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩)

def exact106481RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩]

theorem exact106481RawTermsValid :
    exact106481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20742⟩⟩) exact106481RawTerms .large 106479 .exactZero (none)

def event106482 : Event := .preFoldPolynomial 106481 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩] .exactZero none

def exact106483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20741⟩⟩]⟩, (1)⟩]

def event106483 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20742⟩⟩) 106482 exact106483RawTerms .large 106479 .exactZero (none)

def event106484 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26962⟩⟩)

def event106485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event106486 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event106487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event106488 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event106489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 106488

def event106490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 106486

def event106491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 106489 .coefficient) (.value (.predecessor 1 106490 .coefficient)))

def event106492 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event106493 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11121⟩⟩) 0 ⟨5503⟩ 106492

def event106494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11121⟩⟩) (.authority (.programFamilyFact))

def exact106495RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩], []⟩, (1)⟩]

theorem exact106495RawTermsValid :
    exact106495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106495 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11121⟩⟩) exact106495RawTerms (.finite 6) 106494 .exactZero (none)

def eventLeaf6640 : Array AnnotatedEvent := #[
  { event := event106240
    frameStart := 0 },
  { event := event106241
    frameStart := 0 },
  { event := event106242
    frameStart := 0 },
  { event := event106243
    frameStart := 0 },
  { event := event106244
    frameStart := 0 },
  { event := event106245
    frameStart := 0 },
  { event := event106246
    frameStart := 0 },
  { event := event106247
    frameStart := 0 },
  { event := event106248
    frameStart := 0 },
  { event := event106249
    frameStart := 0 },
  { event := event106250
    frameStart := 0 },
  { event := event106251
    frameStart := 0 },
  { event := event106252
    frameStart := 0 },
  { event := event106253
    frameStart := 0 },
  { event := event106254
    frameStart := 106254 },
  { event := event106255
    frameStart := 106254 }
]

def eventLeaf6641 : Array AnnotatedEvent := #[
  { event := event106256
    frameStart := 106254 },
  { event := event106257
    frameStart := 106254 },
  { event := event106258
    frameStart := 106254 },
  { event := event106259
    frameStart := 106254 },
  { event := event106260
    frameStart := 106254 },
  { event := event106261
    frameStart := 106254 },
  { event := event106262
    frameStart := 106254 },
  { event := event106263
    frameStart := 106254 },
  { event := event106264
    frameStart := 106254 },
  { event := event106265
    frameStart := 106254 },
  { event := event106266
    frameStart := 106254 },
  { event := event106267
    frameStart := 106254 },
  { event := event106268
    frameStart := 106254 },
  { event := event106269
    frameStart := 106254 },
  { event := event106270
    frameStart := 106254 },
  { event := event106271
    frameStart := 106254 }
]

def eventLeaf6642 : Array AnnotatedEvent := #[
  { event := event106272
    frameStart := 106254 },
  { event := event106273
    frameStart := 106254 },
  { event := event106274
    frameStart := 106254 },
  { event := event106275
    frameStart := 106254 },
  { event := event106276
    frameStart := 106254 },
  { event := event106277
    frameStart := 106254 },
  { event := event106278
    frameStart := 106254 },
  { event := event106279
    frameStart := 106254 },
  { event := event106280
    frameStart := 106254 },
  { event := event106281
    frameStart := 106254 },
  { event := event106282
    frameStart := 106254 },
  { event := event106283
    frameStart := 106254 },
  { event := event106284
    frameStart := 106254 },
  { event := event106285
    frameStart := 106254 },
  { event := event106286
    frameStart := 106254 },
  { event := event106287
    frameStart := 106254 }
]

def eventLeaf6643 : Array AnnotatedEvent := #[
  { event := event106288
    frameStart := 106254 },
  { event := event106289
    frameStart := 106254 },
  { event := event106290
    frameStart := 106254 },
  { event := event106291
    frameStart := 106254 },
  { event := event106292
    frameStart := 106254 },
  { event := event106293
    frameStart := 106254 },
  { event := event106294
    frameStart := 106254 },
  { event := event106295
    frameStart := 106254 },
  { event := event106296
    frameStart := 106296 },
  { event := event106297
    frameStart := 106296 },
  { event := event106298
    frameStart := 106296 },
  { event := event106299
    frameStart := 106296 },
  { event := event106300
    frameStart := 106296 },
  { event := event106301
    frameStart := 106296 },
  { event := event106302
    frameStart := 106296 },
  { event := event106303
    frameStart := 106296 }
]

def eventLeaf6644 : Array AnnotatedEvent := #[
  { event := event106304
    frameStart := 106296 },
  { event := event106305
    frameStart := 106296 },
  { event := event106306
    frameStart := 106296 },
  { event := event106307
    frameStart := 106296 },
  { event := event106308
    frameStart := 106296 },
  { event := event106309
    frameStart := 106296 },
  { event := event106310
    frameStart := 106296 },
  { event := event106311
    frameStart := 106296 },
  { event := event106312
    frameStart := 106296 },
  { event := event106313
    frameStart := 106296 },
  { event := event106314
    frameStart := 106296 },
  { event := event106315
    frameStart := 106296 },
  { event := event106316
    frameStart := 106296 },
  { event := event106317
    frameStart := 106296 },
  { event := event106318
    frameStart := 106296 },
  { event := event106319
    frameStart := 106296 }
]

def eventLeaf6645 : Array AnnotatedEvent := #[
  { event := event106320
    frameStart := 106296 },
  { event := event106321
    frameStart := 106296 },
  { event := event106322
    frameStart := 106296 },
  { event := event106323
    frameStart := 106296 },
  { event := event106324
    frameStart := 106296 },
  { event := event106325
    frameStart := 106296 },
  { event := event106326
    frameStart := 106296 },
  { event := event106327
    frameStart := 106296 },
  { event := event106328
    frameStart := 106296 },
  { event := event106329
    frameStart := 106296 },
  { event := event106330
    frameStart := 106296 },
  { event := event106331
    frameStart := 106296 },
  { event := event106332
    frameStart := 106296 },
  { event := event106333
    frameStart := 106296 },
  { event := event106334
    frameStart := 106296 },
  { event := event106335
    frameStart := 106296 }
]

def eventLeaf6646 : Array AnnotatedEvent := #[
  { event := event106336
    frameStart := 106296 },
  { event := event106337
    frameStart := 106296 },
  { event := event106338
    frameStart := 106296 },
  { event := event106339
    frameStart := 106296 },
  { event := event106340
    frameStart := 106296 },
  { event := event106341
    frameStart := 106296 },
  { event := event106342
    frameStart := 106296 },
  { event := event106343
    frameStart := 106296 },
  { event := event106344
    frameStart := 106296 },
  { event := event106345
    frameStart := 106296 },
  { event := event106346
    frameStart := 106296 },
  { event := event106347
    frameStart := 106296 },
  { event := event106348
    frameStart := 106296 },
  { event := event106349
    frameStart := 106296 },
  { event := event106350
    frameStart := 106296 },
  { event := event106351
    frameStart := 106296 }
]

def eventLeaf6647 : Array AnnotatedEvent := #[
  { event := event106352
    frameStart := 106296 },
  { event := event106353
    frameStart := 106296 },
  { event := event106354
    frameStart := 106296 },
  { event := event106355
    frameStart := 106296 },
  { event := event106356
    frameStart := 106296 },
  { event := event106357
    frameStart := 106296 },
  { event := event106358
    frameStart := 106296 },
  { event := event106359
    frameStart := 106296 },
  { event := event106360
    frameStart := 106296 },
  { event := event106361
    frameStart := 106296 },
  { event := event106362
    frameStart := 106296 },
  { event := event106363
    frameStart := 106296 },
  { event := event106364
    frameStart := 106296 },
  { event := event106365
    frameStart := 106296 },
  { event := event106366
    frameStart := 106296 },
  { event := event106367
    frameStart := 106296 }
]

def eventLeaf6648 : Array AnnotatedEvent := #[
  { event := event106368
    frameStart := 106296 },
  { event := event106369
    frameStart := 106296 },
  { event := event106370
    frameStart := 106296 },
  { event := event106371
    frameStart := 106296 },
  { event := event106372
    frameStart := 106296 },
  { event := event106373
    frameStart := 106296 },
  { event := event106374
    frameStart := 106296 },
  { event := event106375
    frameStart := 106296 },
  { event := event106376
    frameStart := 106296 },
  { event := event106377
    frameStart := 106296 },
  { event := event106378
    frameStart := 106296 },
  { event := event106379
    frameStart := 106296 },
  { event := event106380
    frameStart := 106296 },
  { event := event106381
    frameStart := 106296 },
  { event := event106382
    frameStart := 106296 },
  { event := event106383
    frameStart := 106296 }
]

def eventLeaf6649 : Array AnnotatedEvent := #[
  { event := event106384
    frameStart := 106296 },
  { event := event106385
    frameStart := 106296 },
  { event := event106386
    frameStart := 106296 },
  { event := event106387
    frameStart := 106296 },
  { event := event106388
    frameStart := 0 },
  { event := event106389
    frameStart := 0 },
  { event := event106390
    frameStart := 0 },
  { event := event106391
    frameStart := 0 },
  { event := event106392
    frameStart := 0 },
  { event := event106393
    frameStart := 0 },
  { event := event106394
    frameStart := 0 },
  { event := event106395
    frameStart := 0 },
  { event := event106396
    frameStart := 0 },
  { event := event106397
    frameStart := 0 },
  { event := event106398
    frameStart := 0 },
  { event := event106399
    frameStart := 0 }
]

def eventLeaf6650 : Array AnnotatedEvent := #[
  { event := event106400
    frameStart := 0 },
  { event := event106401
    frameStart := 0 },
  { event := event106402
    frameStart := 0 },
  { event := event106403
    frameStart := 0 },
  { event := event106404
    frameStart := 0 },
  { event := event106405
    frameStart := 0 },
  { event := event106406
    frameStart := 0 },
  { event := event106407
    frameStart := 0 },
  { event := event106408
    frameStart := 0 },
  { event := event106409
    frameStart := 0 },
  { event := event106410
    frameStart := 0 },
  { event := event106411
    frameStart := 0 },
  { event := event106412
    frameStart := 0 },
  { event := event106413
    frameStart := 0 },
  { event := event106414
    frameStart := 0 },
  { event := event106415
    frameStart := 0 }
]

def eventLeaf6651 : Array AnnotatedEvent := #[
  { event := event106416
    frameStart := 0 },
  { event := event106417
    frameStart := 0 },
  { event := event106418
    frameStart := 0 },
  { event := event106419
    frameStart := 0 },
  { event := event106420
    frameStart := 0 },
  { event := event106421
    frameStart := 0 },
  { event := event106422
    frameStart := 0 },
  { event := event106423
    frameStart := 0 },
  { event := event106424
    frameStart := 0 },
  { event := event106425
    frameStart := 0 },
  { event := event106426
    frameStart := 0 },
  { event := event106427
    frameStart := 0 },
  { event := event106428
    frameStart := 0 },
  { event := event106429
    frameStart := 0 },
  { event := event106430
    frameStart := 0 },
  { event := event106431
    frameStart := 0 }
]

def eventLeaf6652 : Array AnnotatedEvent := #[
  { event := event106432
    frameStart := 0 },
  { event := event106433
    frameStart := 0 },
  { event := event106434
    frameStart := 0 },
  { event := event106435
    frameStart := 0 },
  { event := event106436
    frameStart := 0 },
  { event := event106437
    frameStart := 0 },
  { event := event106438
    frameStart := 0 },
  { event := event106439
    frameStart := 0 },
  { event := event106440
    frameStart := 0 },
  { event := event106441
    frameStart := 0 },
  { event := event106442
    frameStart := 106442 },
  { event := event106443
    frameStart := 106442 },
  { event := event106444
    frameStart := 106442 },
  { event := event106445
    frameStart := 106442 },
  { event := event106446
    frameStart := 106442 },
  { event := event106447
    frameStart := 106442 }
]

def eventLeaf6653 : Array AnnotatedEvent := #[
  { event := event106448
    frameStart := 106442 },
  { event := event106449
    frameStart := 106442 },
  { event := event106450
    frameStart := 106442 },
  { event := event106451
    frameStart := 106442 },
  { event := event106452
    frameStart := 106442 },
  { event := event106453
    frameStart := 106442 },
  { event := event106454
    frameStart := 106442 },
  { event := event106455
    frameStart := 106442 },
  { event := event106456
    frameStart := 106442 },
  { event := event106457
    frameStart := 106442 },
  { event := event106458
    frameStart := 106442 },
  { event := event106459
    frameStart := 106442 },
  { event := event106460
    frameStart := 106442 },
  { event := event106461
    frameStart := 106442 },
  { event := event106462
    frameStart := 106442 },
  { event := event106463
    frameStart := 106442 }
]

def eventLeaf6654 : Array AnnotatedEvent := #[
  { event := event106464
    frameStart := 106442 },
  { event := event106465
    frameStart := 106442 },
  { event := event106466
    frameStart := 106442 },
  { event := event106467
    frameStart := 106442 },
  { event := event106468
    frameStart := 106442 },
  { event := event106469
    frameStart := 106442 },
  { event := event106470
    frameStart := 106442 },
  { event := event106471
    frameStart := 106442 },
  { event := event106472
    frameStart := 106442 },
  { event := event106473
    frameStart := 106442 },
  { event := event106474
    frameStart := 106442 },
  { event := event106475
    frameStart := 106442 },
  { event := event106476
    frameStart := 106442 },
  { event := event106477
    frameStart := 106442 },
  { event := event106478
    frameStart := 106442 },
  { event := event106479
    frameStart := 106442 }
]

def eventLeaf6655 : Array AnnotatedEvent := #[
  { event := event106480
    frameStart := 106442 },
  { event := event106481
    frameStart := 106442 },
  { event := event106482
    frameStart := 106442 },
  { event := event106483
    frameStart := 106442 },
  { event := event106484
    frameStart := 106484 },
  { event := event106485
    frameStart := 106484 },
  { event := event106486
    frameStart := 106484 },
  { event := event106487
    frameStart := 106484 },
  { event := event106488
    frameStart := 106484 },
  { event := event106489
    frameStart := 106484 },
  { event := event106490
    frameStart := 106484 },
  { event := event106491
    frameStart := 106484 },
  { event := event106492
    frameStart := 106484 },
  { event := event106493
    frameStart := 106484 },
  { event := event106494
    frameStart := 106484 },
  { event := event106495
    frameStart := 106484 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events415
