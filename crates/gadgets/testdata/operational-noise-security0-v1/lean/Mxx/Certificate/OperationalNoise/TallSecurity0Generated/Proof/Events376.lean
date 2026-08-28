import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events376

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event96256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 0 ⟨9910⟩ 96255

def event96257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 1 ⟨12542⟩ 96252

def event96258 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12543⟩⟩) (.product (.predecessor 0 96256 .coefficient) (.predecessor 1 96257 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12543⟩⟩, .operator (⟨96255, 0⟩, ⟨96252, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩)

def exact96260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩]

theorem exact96260RawTermsValid :
    exact96260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12543⟩⟩) exact96260RawTerms (.finite 1764) 96258 .exactZero (none)

def event96261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12544⟩⟩) 0 ⟨12543⟩ 96260

def event96262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.identity (.predecessor 0 96261 .coefficient))

def event96263 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.finite 1764)

def event96264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23241⟩⟩) 0 ⟨12544⟩ 96263

def event96265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23241⟩⟩) (.authority (.programFamilyFact))

def event96266 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23241⟩⟩) (.finite 3720)

def event96267 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event96268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23242⟩⟩) 0 ⟨6689⟩ 96267

def event96269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23242⟩⟩) 1 ⟨23241⟩ 96266

def event96270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23242⟩⟩) (.authority (.operator))

def exact96271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩]

theorem exact96271RawTermsValid :
    exact96271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23242⟩⟩) exact96271RawTerms .large 96270 .exactZero (none)

def event96272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25437⟩⟩) 0 ⟨23242⟩ 96271

def event96273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25437⟩⟩) (.authority (.operator))

def exact96274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (1)⟩]

theorem exact96274RawTermsValid :
    exact96274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96274 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25437⟩⟩) exact96274RawTerms (.finite 8192) 96273 .exactZero (none)

def event96275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event96276 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event96277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12654⟩⟩) 0 ⟨12544⟩ 96263

def event96278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12654⟩⟩) 1 ⟨110⟩ 96276

def event96279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12654⟩⟩) (.sum [.predecessor 0 96277 .coefficient, .predecessor 1 96278 .coefficient])

def event96280 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12654⟩⟩) (.finite 1764)

def event96281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12655⟩⟩) 0 ⟨12654⟩ 96280

def event96282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12655⟩⟩) (.identity (.predecessor 0 96281 .coefficient))

def exact96283RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩]

theorem exact96283RawTermsValid :
    exact96283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96283 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12655⟩⟩) exact96283RawTerms (.finite 1764) 96282 .exactZero (none)

def event96284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact96285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96285RawTermsValid :
    exact96285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96285 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact96285RawTerms .large 96284 .exactZero (none)

def event96286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12656⟩⟩) 0 ⟨6544⟩ 96285

def event96287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12656⟩⟩) 1 ⟨12655⟩ 96283

def event96288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12656⟩⟩) (.product (.predecessor 0 96286 .coefficient) (.predecessor 1 96287 .coefficient) (⟨false, false, none, none, none⟩))

def event96289 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12656⟩⟩, .operator (⟨96285, 0⟩, ⟨96283, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96290RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96290RawTermsValid :
    exact96290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96290 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12656⟩⟩) exact96290RawTerms .large 96288 .exactZero (none)

def event96291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event96292 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event96293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 96267

def event96294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact96295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact96295RawTermsValid :
    exact96295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96295 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact96295RawTerms .large 96294 .exactZero (none)

def event96296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6786⟩⟩) 0 ⟨6757⟩ 96295

def event96297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6786⟩⟩) (.identity (.predecessor 0 96296 .coefficient))

def exact96298RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩, (1)⟩]

theorem exact96298RawTermsValid :
    exact96298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96298 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6786⟩⟩) exact96298RawTerms .large 96297 .exactZero (none)

def event96299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7870⟩⟩) 0 ⟨6786⟩ 96298

def event96300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7870⟩⟩) (.authority (.operator))

def exact96301RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact96301RawTermsValid :
    exact96301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7870⟩⟩) exact96301RawTerms (.finite 8192) 96300 .exactZero (none)

def event96302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 0 ⟨7870⟩ 96301

def event96303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 1 ⟨2348⟩ 96292

def event96304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7871⟩⟩) (.scale (.predecessor 0 96302 .coefficient) (.value (.predecessor 1 96303 .coefficient)))

def exact96305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact96305RawTermsValid :
    exact96305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7871⟩⟩) exact96305RawTerms (.finite 8192) 96304 .exactZero (none)

def event96306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6766⟩⟩) 0 ⟨6757⟩ 96295

def event96307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6766⟩⟩) (.identity (.predecessor 0 96306 .coefficient))

def exact96308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩]⟩, (1)⟩]

theorem exact96308RawTermsValid :
    exact96308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96308 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6766⟩⟩) exact96308RawTerms .large 96307 .exactZero (none)

def event96309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 0 ⟨6766⟩ 96308

def event96310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 1 ⟨7871⟩ 96305

def event96311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7872⟩⟩) (.product (.predecessor 0 96309 .coefficient) (.predecessor 1 96310 .coefficient) (⟨false, false, none, none, none⟩))

def event96312 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7872⟩⟩, .operator (⟨96308, 0⟩, ⟨96305, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩)

def exact96313RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact96313RawTermsValid :
    exact96313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96313 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7872⟩⟩) exact96313RawTerms .large 96311 .exactZero (none)

def event96314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12657⟩⟩) 0 ⟨7872⟩ 96313

def event96315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12657⟩⟩) 1 ⟨12656⟩ 96290

def event96316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12657⟩⟩) (.sum [.predecessor 0 96314 .coefficient, .predecessor 1 96315 .coefficient])

def exact96317RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96317RawTermsValid :
    exact96317RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96317 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12657⟩⟩) exact96317RawTerms .large 96316 .exactZero (none)

def event96318 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25440⟩⟩) 0 ⟨12657⟩ 96317

def event96319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25440⟩⟩) 1 ⟨25437⟩ 96274

def event96320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25440⟩⟩) (.product (.predecessor 0 96318 .coefficient) (.predecessor 1 96319 .coefficient) (⟨false, false, none, none, none⟩))

def event96321 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25440⟩⟩, .operator (⟨96317, 0⟩, ⟨96274, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (1)⟩)

def event96322 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25440⟩⟩, .operator (⟨96317, 1⟩, ⟨96274, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩)

def event96323 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25440⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25437⟩⟩) ⟨23242⟩ 96271)

def event96324 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25440⟩⟩, .relation 96323 0, ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (-1)⟩)

def exact96325RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (-1)⟩]

theorem exact96325RawTermsValid :
    exact96325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25440⟩⟩) exact96325RawTerms .large 96320 .exactZero (none)

def event96326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16539⟩⟩) 0 ⟨12544⟩ 96263

def event96327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16539⟩⟩) (.authority (.programFamilyFact))

def exact96328RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], []⟩, (1)⟩]

theorem exact96328RawTermsValid :
    exact96328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16539⟩⟩) exact96328RawTerms (.finite 42) 96327 .exactZero (none)

def event96329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16541⟩⟩) 0 ⟨6544⟩ 96285

def event96330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16541⟩⟩) 1 ⟨16539⟩ 96328

def event96331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16541⟩⟩) (.product (.predecessor 0 96329 .coefficient) (.predecessor 1 96330 .coefficient) (⟨false, true, none, none, some 1⟩))

def event96332 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16541⟩⟩, .operator (⟨96285, 0⟩, ⟨96328, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96333RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96333RawTermsValid :
    exact96333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16541⟩⟩) exact96333RawTerms .large 96331 .exactZero (none)

def event96334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6703⟩⟩) 0 ⟨6689⟩ 96267

def event96335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6703⟩⟩) (.authority (.operator))

def exact96336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩]

theorem exact96336RawTermsValid :
    exact96336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6703⟩⟩) exact96336RawTerms .large 96335 .exactZero (none)

def event96337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16542⟩⟩) 0 ⟨6703⟩ 96336

def event96338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16542⟩⟩) 1 ⟨16541⟩ 96333

def event96339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16542⟩⟩) (.sum [.predecessor 0 96337 .coefficient, .predecessor 1 96338 .coefficient])

def exact96340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96340RawTermsValid :
    exact96340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96340 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16542⟩⟩) exact96340RawTerms .large 96339 .exactZero (none)

def event96341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25441⟩⟩) 0 ⟨16542⟩ 96340

def event96342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25441⟩⟩) 1 ⟨25440⟩ 96325

def event96343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25441⟩⟩) (.sum [.predecessor 0 96341 .coefficient, .predecessor 1 96342 .coefficient])

def exact96344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96344RawTermsValid :
    exact96344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25441⟩⟩) exact96344RawTerms .large 96343 .exactZero (none)

def event96345 : Event := .preFoldPolynomial 96344 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact96346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event96346 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25441⟩⟩) 96345 exact96346RawTerms .large 96343 .exactZero (none)

def event96347 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12544⟩⟩) ⟨⟨116⟩, ⟨21⟩, ⟨109⟩⟩ ⟨96205, 96347⟩

def event96348 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19952⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19949⟩⟩]⟩) (1) 0 2 (.universal 96347 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19949⟩⟩]⟩) (none) 96346)

def event96349 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19952⟩⟩, .relation 96348 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩)

def event96350 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19952⟩⟩, .relation 96348 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩)

def event96351 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19952⟩⟩, .relation 96348 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩)

def event96352 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19952⟩⟩, .relation 96348 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact96353RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96353RawTermsValid :
    exact96353RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96353 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19952⟩⟩) exact96353RawTerms .large 96201 (.finite 1811303510016) (some (96203))

def event96354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25439⟩⟩) 0 ⟨19952⟩ 96353

def event96355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25439⟩⟩) 1 ⟨25438⟩ 96191

def event96356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25439⟩⟩) (.sum [.predecessor 0 96354 .coefficient, .predecessor 1 96355 .coefficient])

def event96357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25439⟩⟩, .operator (⟨96353, 2⟩, ⟨96191, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨23242⟩⟩]⟩, (-1)⟩)

def event96358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25439⟩⟩, .operator (⟨96353, 1⟩, ⟨96191, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩, (1)⟩)

def event96359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25439⟩⟩) (.sum [.result 96353 .summary, .result 96191 .summary])

def exact96360RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96360RawTermsValid :
    exact96360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96360 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25439⟩⟩) exact96360RawTerms .large 96356 (.finite 352134001995776) (some (96359))

def event96361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29135⟩⟩) 0 ⟨25439⟩ 96360

def event96362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29135⟩⟩) 1 ⟨29133⟩ 96107

def event96363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29135⟩⟩) (.product (.predecessor 0 96361 .coefficient) (.predecessor 1 96362 .coefficient) (⟨false, false, none, none, none⟩))

def event96364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29135⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) [⟨.result 96107 .coefficient, false, none⟩])

def event96365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29135⟩⟩) (.product (.result 96360 .summary) (.transfer 96364) (⟨false, false, none, none, none⟩))

def event96366 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29135⟩⟩, .operator (⟨96360, 0⟩, ⟨96107, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (1)⟩)

def event96367 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29135⟩⟩, .operator (⟨96360, 1⟩, ⟨96107, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (-1)⟩)

def event96368 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29135⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29133⟩⟩) ⟨24531⟩ 96104)

def event96369 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29135⟩⟩, .relation 96368 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩, (-1)⟩)

def exact96370RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩, (-1)⟩]

theorem exact96370RawTermsValid :
    exact96370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29135⟩⟩) exact96370RawTerms .large 96363 (.finite 1292337421468529852416) (some (96365))

def event96371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22253⟩⟩) 0 ⟨16540⟩ 4675

def event96372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22253⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact96373RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩]

theorem exact96373RawTermsValid :
    exact96373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22253⟩⟩) exact96373RawTerms (.finite 136065468) 96372 .exactZero (none)

def event96374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22255⟩⟩) 0 ⟨22253⟩ 96373

def event96375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22255⟩⟩) 1 ⟨2348⟩ 4

def event96376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22255⟩⟩) (.scale (.predecessor 0 96374 .coefficient) (.value (.predecessor 1 96375 .coefficient)))

def exact96377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩]

theorem exact96377RawTermsValid :
    exact96377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22255⟩⟩) exact96377RawTerms (.finite 136065468) 96376 .exactZero (none)

def event96378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22256⟩⟩) 0 ⟨5509⟩ 94462

def event96379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22256⟩⟩) 1 ⟨22255⟩ 96377

def event96380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22256⟩⟩) (.product (.predecessor 0 96378 .coefficient) (.predecessor 1 96379 .coefficient) (⟨false, false, none, none, none⟩))

def event96381 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22256⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩) [⟨.result 96373 .coefficient, false, none⟩])

def event96382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22256⟩⟩) (.product (.result 94462 .summary) (.transfer 96381) (⟨false, false, none, none, none⟩))

def event96383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22256⟩⟩, .operator (⟨94462, 0⟩, ⟨96377, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩)

def event96384 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22254⟩⟩)

def event96385 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event96386 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event96387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event96388 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event96389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 96388

def event96390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 96386

def event96391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 96389 .coefficient) (.value (.predecessor 1 96390 .coefficient)))

def event96392 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event96393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12542⟩⟩) 0 ⟨5503⟩ 96392

def event96394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12542⟩⟩) (.authority (.programFamilyFact))

def exact96395RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩]

theorem exact96395RawTermsValid :
    exact96395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12542⟩⟩) exact96395RawTerms (.finite 42) 96394 .exactZero (none)

def event96396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9910⟩⟩) 0 ⟨5503⟩ 96392

def event96397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9910⟩⟩) (.authority (.programFamilyFact))

def exact96398RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩], []⟩, (1)⟩]

theorem exact96398RawTermsValid :
    exact96398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96398 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9910⟩⟩) exact96398RawTerms (.finite 42) 96397 .exactZero (none)

def event96399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 0 ⟨9910⟩ 96398

def event96400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 1 ⟨12542⟩ 96395

def event96401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12543⟩⟩) (.product (.predecessor 0 96399 .coefficient) (.predecessor 1 96400 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12543⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩) [⟨.result 96398 .coefficient, true, some 1⟩, ⟨.result 96395 .coefficient, true, some 1⟩])

def event96403 : Event := .survivorFold (1) 96402

def exact96404RawTerms : List Term := []

theorem exact96404RawTermsValid :
    exact96404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12543⟩⟩) exact96404RawTerms (.finite 1764) 96401 (.finite 1764) (some (96402))

def event96405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12544⟩⟩) 0 ⟨12543⟩ 96404

def event96406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.identity (.predecessor 0 96405 .coefficient))

def event96407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.finite 1764)

def event96408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16539⟩⟩) 0 ⟨12544⟩ 96407

def event96409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16539⟩⟩) (.authority (.programFamilyFact))

def exact96410RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], []⟩, (1)⟩]

theorem exact96410RawTermsValid :
    exact96410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16539⟩⟩) exact96410RawTerms (.finite 42) 96409 .exactZero (none)

def event96411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16540⟩⟩) 0 ⟨16539⟩ 96410

def event96412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16540⟩⟩) (.identity (.predecessor 0 96411 .coefficient))

def event96413 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16540⟩⟩) (.finite 42)

def event96414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22253⟩⟩) 0 ⟨16540⟩ 96413

def event96415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22253⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact96416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩]

theorem exact96416RawTermsValid :
    exact96416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22253⟩⟩) exact96416RawTerms (.finite 136065468) 96415 .exactZero (none)

def event96417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact96418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact96418RawTermsValid :
    exact96418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact96418RawTerms .large 96417 .exactZero (none)

def event96419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22254⟩⟩) 0 ⟨6⟩ 96418

def event96420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22254⟩⟩) 1 ⟨22253⟩ 96416

def event96421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22254⟩⟩) (.product (.predecessor 0 96419 .coefficient) (.predecessor 1 96420 .coefficient) (⟨false, false, none, none, none⟩))

def event96422 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22254⟩⟩, .operator (⟨96418, 0⟩, ⟨96416, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩)

def exact96423RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩]

theorem exact96423RawTermsValid :
    exact96423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96423 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22254⟩⟩) exact96423RawTerms .large 96421 .exactZero (none)

def event96424 : Event := .preFoldPolynomial 96423 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩] .exactZero none

def exact96425RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22253⟩⟩]⟩, (1)⟩]

def event96425 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22254⟩⟩) 96424 exact96425RawTerms .large 96421 .exactZero (none)

def event96426 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29138⟩⟩)

def event96427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event96428 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event96429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event96430 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event96431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 96430

def event96432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 96428

def event96433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 96431 .coefficient) (.value (.predecessor 1 96432 .coefficient)))

def event96434 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event96435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12542⟩⟩) 0 ⟨5503⟩ 96434

def event96436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12542⟩⟩) (.authority (.programFamilyFact))

def exact96437RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩]

theorem exact96437RawTermsValid :
    exact96437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96437 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12542⟩⟩) exact96437RawTerms (.finite 42) 96436 .exactZero (none)

def event96438 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9910⟩⟩) 0 ⟨5503⟩ 96434

def event96439 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9910⟩⟩) (.authority (.programFamilyFact))

def exact96440RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩], []⟩, (1)⟩]

theorem exact96440RawTermsValid :
    exact96440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96440 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9910⟩⟩) exact96440RawTerms (.finite 42) 96439 .exactZero (none)

def event96441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 0 ⟨9910⟩ 96440

def event96442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12543⟩⟩) 1 ⟨12542⟩ 96437

def event96443 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12543⟩⟩) (.product (.predecessor 0 96441 .coefficient) (.predecessor 1 96442 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96444 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12543⟩⟩, .operator (⟨96440, 0⟩, ⟨96437, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩)

def exact96445RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], []⟩, (1)⟩]

theorem exact96445RawTermsValid :
    exact96445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12543⟩⟩) exact96445RawTerms (.finite 1764) 96443 .exactZero (none)

def event96446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12544⟩⟩) 0 ⟨12543⟩ 96445

def event96447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.identity (.predecessor 0 96446 .coefficient))

def event96448 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12544⟩⟩) (.finite 1764)

def event96449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16539⟩⟩) 0 ⟨12544⟩ 96448

def event96450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16539⟩⟩) (.authority (.programFamilyFact))

def exact96451RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], []⟩, (1)⟩]

theorem exact96451RawTermsValid :
    exact96451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96451 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16539⟩⟩) exact96451RawTerms (.finite 42) 96450 .exactZero (none)

def event96452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16540⟩⟩) 0 ⟨16539⟩ 96451

def event96453 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16540⟩⟩) (.identity (.predecessor 0 96452 .coefficient))

def event96454 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16540⟩⟩) (.finite 42)

def event96455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24529⟩⟩) 0 ⟨16540⟩ 96454

def event96456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24529⟩⟩) (.authority (.programFamilyFact))

def event96457 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24529⟩⟩) (.finite 3720)

def event96458 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event96459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24531⟩⟩) 0 ⟨6689⟩ 96458

def event96460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24531⟩⟩) 1 ⟨24529⟩ 96457

def event96461 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24531⟩⟩) (.authority (.operator))

def exact96462RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩, (1)⟩]

theorem exact96462RawTermsValid :
    exact96462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24531⟩⟩) exact96462RawTerms .large 96461 .exactZero (none)

def event96463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29133⟩⟩) 0 ⟨24531⟩ 96462

def event96464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29133⟩⟩) (.authority (.operator))

def exact96465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (1)⟩]

theorem exact96465RawTermsValid :
    exact96465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96465 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29133⟩⟩) exact96465RawTerms (.finite 8192) 96464 .exactZero (none)

def event96466 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event96467 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event96468 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16581⟩⟩) 0 ⟨16540⟩ 96454

def event96469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16581⟩⟩) 1 ⟨110⟩ 96467

def event96470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16581⟩⟩) (.sum [.predecessor 0 96468 .coefficient, .predecessor 1 96469 .coefficient])

def event96471 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16581⟩⟩) (.finite 42)

def event96472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16582⟩⟩) 0 ⟨16581⟩ 96471

def event96473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16582⟩⟩) (.identity (.predecessor 0 96472 .coefficient))

def exact96474RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], []⟩, (1)⟩]

theorem exact96474RawTermsValid :
    exact96474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16582⟩⟩) exact96474RawTerms (.finite 42) 96473 .exactZero (none)

def event96475 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact96476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96476RawTermsValid :
    exact96476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact96476RawTerms .large 96475 .exactZero (none)

def event96477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16583⟩⟩) 0 ⟨6544⟩ 96476

def event96478 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16583⟩⟩) 1 ⟨16582⟩ 96474

def event96479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16583⟩⟩) (.product (.predecessor 0 96477 .coefficient) (.predecessor 1 96478 .coefficient) (⟨false, false, none, none, none⟩))

def event96480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16583⟩⟩, .operator (⟨96476, 0⟩, ⟨96474, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96481RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96481RawTermsValid :
    exact96481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16583⟩⟩) exact96481RawTerms .large 96479 .exactZero (none)

def event96482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6703⟩⟩) 0 ⟨6689⟩ 96458

def event96483 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6703⟩⟩) (.authority (.operator))

def exact96484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩]

theorem exact96484RawTermsValid :
    exact96484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6703⟩⟩) exact96484RawTerms .large 96483 .exactZero (none)

def event96485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16584⟩⟩) 0 ⟨6703⟩ 96484

def event96486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16584⟩⟩) 1 ⟨16583⟩ 96481

def event96487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16584⟩⟩) (.sum [.predecessor 0 96485 .coefficient, .predecessor 1 96486 .coefficient])

def exact96488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96488RawTermsValid :
    exact96488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16584⟩⟩) exact96488RawTerms .large 96487 .exactZero (none)

def event96489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29134⟩⟩) 0 ⟨16584⟩ 96488

def event96490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29134⟩⟩) 1 ⟨29133⟩ 96465

def event96491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29134⟩⟩) (.product (.predecessor 0 96489 .coefficient) (.predecessor 1 96490 .coefficient) (⟨false, false, none, none, none⟩))

def event96492 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29134⟩⟩, .operator (⟨96488, 0⟩, ⟨96465, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (1)⟩)

def event96493 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29134⟩⟩, .operator (⟨96488, 1⟩, ⟨96465, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (-1)⟩)

def event96494 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29134⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29133⟩⟩) ⟨24531⟩ 96462)

def event96495 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29134⟩⟩, .relation 96494 0, ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩, (-1)⟩)

def exact96496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29133⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24531⟩⟩]⟩, (-1)⟩]

theorem exact96496RawTermsValid :
    exact96496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29134⟩⟩) exact96496RawTerms .large 96491 .exactZero (none)

def event96497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18198⟩⟩) 0 ⟨16540⟩ 96454

def event96498 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18198⟩⟩) (.authority (.programFamilyFact))

def exact96499RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], []⟩, (1)⟩]

theorem exact96499RawTermsValid :
    exact96499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96499 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18198⟩⟩) exact96499RawTerms (.finite 63) 96498 .exactZero (none)

def event96500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18199⟩⟩) 0 ⟨6544⟩ 96476

def event96501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18199⟩⟩) 1 ⟨18198⟩ 96499

def event96502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18199⟩⟩) (.product (.predecessor 0 96500 .coefficient) (.predecessor 1 96501 .coefficient) (⟨false, true, none, none, some 1⟩))

def event96503 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18199⟩⟩, .operator (⟨96476, 0⟩, ⟨96499, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact96504RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact96504RawTermsValid :
    exact96504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18199⟩⟩) exact96504RawTerms .large 96502 .exactZero (none)

def event96505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6735⟩⟩) 0 ⟨6689⟩ 96458

def event96506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6735⟩⟩) (.authority (.operator))

def exact96507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩]

theorem exact96507RawTermsValid :
    exact96507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96507 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6735⟩⟩) exact96507RawTerms .large 96506 .exactZero (none)

def event96508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18200⟩⟩) 0 ⟨6735⟩ 96507

def event96509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18200⟩⟩) 1 ⟨18199⟩ 96504

def event96510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18200⟩⟩) (.sum [.predecessor 0 96508 .coefficient, .predecessor 1 96509 .coefficient])

def exact96511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact96511RawTermsValid :
    exact96511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96511 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18200⟩⟩) exact96511RawTerms .large 96510 .exactZero (none)

def eventLeaf6016 : Array AnnotatedEvent := #[
  { event := event96256
    frameStart := 96241 },
  { event := event96257
    frameStart := 96241 },
  { event := event96258
    frameStart := 96241 },
  { event := event96259
    frameStart := 96241 },
  { event := event96260
    frameStart := 96241 },
  { event := event96261
    frameStart := 96241 },
  { event := event96262
    frameStart := 96241 },
  { event := event96263
    frameStart := 96241 },
  { event := event96264
    frameStart := 96241 },
  { event := event96265
    frameStart := 96241 },
  { event := event96266
    frameStart := 96241 },
  { event := event96267
    frameStart := 96241 },
  { event := event96268
    frameStart := 96241 },
  { event := event96269
    frameStart := 96241 },
  { event := event96270
    frameStart := 96241 },
  { event := event96271
    frameStart := 96241 }
]

def eventLeaf6017 : Array AnnotatedEvent := #[
  { event := event96272
    frameStart := 96241 },
  { event := event96273
    frameStart := 96241 },
  { event := event96274
    frameStart := 96241 },
  { event := event96275
    frameStart := 96241 },
  { event := event96276
    frameStart := 96241 },
  { event := event96277
    frameStart := 96241 },
  { event := event96278
    frameStart := 96241 },
  { event := event96279
    frameStart := 96241 },
  { event := event96280
    frameStart := 96241 },
  { event := event96281
    frameStart := 96241 },
  { event := event96282
    frameStart := 96241 },
  { event := event96283
    frameStart := 96241 },
  { event := event96284
    frameStart := 96241 },
  { event := event96285
    frameStart := 96241 },
  { event := event96286
    frameStart := 96241 },
  { event := event96287
    frameStart := 96241 }
]

def eventLeaf6018 : Array AnnotatedEvent := #[
  { event := event96288
    frameStart := 96241 },
  { event := event96289
    frameStart := 96241 },
  { event := event96290
    frameStart := 96241 },
  { event := event96291
    frameStart := 96241 },
  { event := event96292
    frameStart := 96241 },
  { event := event96293
    frameStart := 96241 },
  { event := event96294
    frameStart := 96241 },
  { event := event96295
    frameStart := 96241 },
  { event := event96296
    frameStart := 96241 },
  { event := event96297
    frameStart := 96241 },
  { event := event96298
    frameStart := 96241 },
  { event := event96299
    frameStart := 96241 },
  { event := event96300
    frameStart := 96241 },
  { event := event96301
    frameStart := 96241 },
  { event := event96302
    frameStart := 96241 },
  { event := event96303
    frameStart := 96241 }
]

def eventLeaf6019 : Array AnnotatedEvent := #[
  { event := event96304
    frameStart := 96241 },
  { event := event96305
    frameStart := 96241 },
  { event := event96306
    frameStart := 96241 },
  { event := event96307
    frameStart := 96241 },
  { event := event96308
    frameStart := 96241 },
  { event := event96309
    frameStart := 96241 },
  { event := event96310
    frameStart := 96241 },
  { event := event96311
    frameStart := 96241 },
  { event := event96312
    frameStart := 96241 },
  { event := event96313
    frameStart := 96241 },
  { event := event96314
    frameStart := 96241 },
  { event := event96315
    frameStart := 96241 },
  { event := event96316
    frameStart := 96241 },
  { event := event96317
    frameStart := 96241 },
  { event := event96318
    frameStart := 96241 },
  { event := event96319
    frameStart := 96241 }
]

def eventLeaf6020 : Array AnnotatedEvent := #[
  { event := event96320
    frameStart := 96241 },
  { event := event96321
    frameStart := 96241 },
  { event := event96322
    frameStart := 96241 },
  { event := event96323
    frameStart := 96241 },
  { event := event96324
    frameStart := 96241 },
  { event := event96325
    frameStart := 96241 },
  { event := event96326
    frameStart := 96241 },
  { event := event96327
    frameStart := 96241 },
  { event := event96328
    frameStart := 96241 },
  { event := event96329
    frameStart := 96241 },
  { event := event96330
    frameStart := 96241 },
  { event := event96331
    frameStart := 96241 },
  { event := event96332
    frameStart := 96241 },
  { event := event96333
    frameStart := 96241 },
  { event := event96334
    frameStart := 96241 },
  { event := event96335
    frameStart := 96241 }
]

def eventLeaf6021 : Array AnnotatedEvent := #[
  { event := event96336
    frameStart := 96241 },
  { event := event96337
    frameStart := 96241 },
  { event := event96338
    frameStart := 96241 },
  { event := event96339
    frameStart := 96241 },
  { event := event96340
    frameStart := 96241 },
  { event := event96341
    frameStart := 96241 },
  { event := event96342
    frameStart := 96241 },
  { event := event96343
    frameStart := 96241 },
  { event := event96344
    frameStart := 96241 },
  { event := event96345
    frameStart := 96241 },
  { event := event96346
    frameStart := 96241 },
  { event := event96347
    frameStart := 0 },
  { event := event96348
    frameStart := 0 },
  { event := event96349
    frameStart := 0 },
  { event := event96350
    frameStart := 0 },
  { event := event96351
    frameStart := 0 }
]

def eventLeaf6022 : Array AnnotatedEvent := #[
  { event := event96352
    frameStart := 0 },
  { event := event96353
    frameStart := 0 },
  { event := event96354
    frameStart := 0 },
  { event := event96355
    frameStart := 0 },
  { event := event96356
    frameStart := 0 },
  { event := event96357
    frameStart := 0 },
  { event := event96358
    frameStart := 0 },
  { event := event96359
    frameStart := 0 },
  { event := event96360
    frameStart := 0 },
  { event := event96361
    frameStart := 0 },
  { event := event96362
    frameStart := 0 },
  { event := event96363
    frameStart := 0 },
  { event := event96364
    frameStart := 0 },
  { event := event96365
    frameStart := 0 },
  { event := event96366
    frameStart := 0 },
  { event := event96367
    frameStart := 0 }
]

def eventLeaf6023 : Array AnnotatedEvent := #[
  { event := event96368
    frameStart := 0 },
  { event := event96369
    frameStart := 0 },
  { event := event96370
    frameStart := 0 },
  { event := event96371
    frameStart := 0 },
  { event := event96372
    frameStart := 0 },
  { event := event96373
    frameStart := 0 },
  { event := event96374
    frameStart := 0 },
  { event := event96375
    frameStart := 0 },
  { event := event96376
    frameStart := 0 },
  { event := event96377
    frameStart := 0 },
  { event := event96378
    frameStart := 0 },
  { event := event96379
    frameStart := 0 },
  { event := event96380
    frameStart := 0 },
  { event := event96381
    frameStart := 0 },
  { event := event96382
    frameStart := 0 },
  { event := event96383
    frameStart := 0 }
]

def eventLeaf6024 : Array AnnotatedEvent := #[
  { event := event96384
    frameStart := 96384 },
  { event := event96385
    frameStart := 96384 },
  { event := event96386
    frameStart := 96384 },
  { event := event96387
    frameStart := 96384 },
  { event := event96388
    frameStart := 96384 },
  { event := event96389
    frameStart := 96384 },
  { event := event96390
    frameStart := 96384 },
  { event := event96391
    frameStart := 96384 },
  { event := event96392
    frameStart := 96384 },
  { event := event96393
    frameStart := 96384 },
  { event := event96394
    frameStart := 96384 },
  { event := event96395
    frameStart := 96384 },
  { event := event96396
    frameStart := 96384 },
  { event := event96397
    frameStart := 96384 },
  { event := event96398
    frameStart := 96384 },
  { event := event96399
    frameStart := 96384 }
]

def eventLeaf6025 : Array AnnotatedEvent := #[
  { event := event96400
    frameStart := 96384 },
  { event := event96401
    frameStart := 96384 },
  { event := event96402
    frameStart := 96384 },
  { event := event96403
    frameStart := 96384 },
  { event := event96404
    frameStart := 96384 },
  { event := event96405
    frameStart := 96384 },
  { event := event96406
    frameStart := 96384 },
  { event := event96407
    frameStart := 96384 },
  { event := event96408
    frameStart := 96384 },
  { event := event96409
    frameStart := 96384 },
  { event := event96410
    frameStart := 96384 },
  { event := event96411
    frameStart := 96384 },
  { event := event96412
    frameStart := 96384 },
  { event := event96413
    frameStart := 96384 },
  { event := event96414
    frameStart := 96384 },
  { event := event96415
    frameStart := 96384 }
]

def eventLeaf6026 : Array AnnotatedEvent := #[
  { event := event96416
    frameStart := 96384 },
  { event := event96417
    frameStart := 96384 },
  { event := event96418
    frameStart := 96384 },
  { event := event96419
    frameStart := 96384 },
  { event := event96420
    frameStart := 96384 },
  { event := event96421
    frameStart := 96384 },
  { event := event96422
    frameStart := 96384 },
  { event := event96423
    frameStart := 96384 },
  { event := event96424
    frameStart := 96384 },
  { event := event96425
    frameStart := 96384 },
  { event := event96426
    frameStart := 96426 },
  { event := event96427
    frameStart := 96426 },
  { event := event96428
    frameStart := 96426 },
  { event := event96429
    frameStart := 96426 },
  { event := event96430
    frameStart := 96426 },
  { event := event96431
    frameStart := 96426 }
]

def eventLeaf6027 : Array AnnotatedEvent := #[
  { event := event96432
    frameStart := 96426 },
  { event := event96433
    frameStart := 96426 },
  { event := event96434
    frameStart := 96426 },
  { event := event96435
    frameStart := 96426 },
  { event := event96436
    frameStart := 96426 },
  { event := event96437
    frameStart := 96426 },
  { event := event96438
    frameStart := 96426 },
  { event := event96439
    frameStart := 96426 },
  { event := event96440
    frameStart := 96426 },
  { event := event96441
    frameStart := 96426 },
  { event := event96442
    frameStart := 96426 },
  { event := event96443
    frameStart := 96426 },
  { event := event96444
    frameStart := 96426 },
  { event := event96445
    frameStart := 96426 },
  { event := event96446
    frameStart := 96426 },
  { event := event96447
    frameStart := 96426 }
]

def eventLeaf6028 : Array AnnotatedEvent := #[
  { event := event96448
    frameStart := 96426 },
  { event := event96449
    frameStart := 96426 },
  { event := event96450
    frameStart := 96426 },
  { event := event96451
    frameStart := 96426 },
  { event := event96452
    frameStart := 96426 },
  { event := event96453
    frameStart := 96426 },
  { event := event96454
    frameStart := 96426 },
  { event := event96455
    frameStart := 96426 },
  { event := event96456
    frameStart := 96426 },
  { event := event96457
    frameStart := 96426 },
  { event := event96458
    frameStart := 96426 },
  { event := event96459
    frameStart := 96426 },
  { event := event96460
    frameStart := 96426 },
  { event := event96461
    frameStart := 96426 },
  { event := event96462
    frameStart := 96426 },
  { event := event96463
    frameStart := 96426 }
]

def eventLeaf6029 : Array AnnotatedEvent := #[
  { event := event96464
    frameStart := 96426 },
  { event := event96465
    frameStart := 96426 },
  { event := event96466
    frameStart := 96426 },
  { event := event96467
    frameStart := 96426 },
  { event := event96468
    frameStart := 96426 },
  { event := event96469
    frameStart := 96426 },
  { event := event96470
    frameStart := 96426 },
  { event := event96471
    frameStart := 96426 },
  { event := event96472
    frameStart := 96426 },
  { event := event96473
    frameStart := 96426 },
  { event := event96474
    frameStart := 96426 },
  { event := event96475
    frameStart := 96426 },
  { event := event96476
    frameStart := 96426 },
  { event := event96477
    frameStart := 96426 },
  { event := event96478
    frameStart := 96426 },
  { event := event96479
    frameStart := 96426 }
]

def eventLeaf6030 : Array AnnotatedEvent := #[
  { event := event96480
    frameStart := 96426 },
  { event := event96481
    frameStart := 96426 },
  { event := event96482
    frameStart := 96426 },
  { event := event96483
    frameStart := 96426 },
  { event := event96484
    frameStart := 96426 },
  { event := event96485
    frameStart := 96426 },
  { event := event96486
    frameStart := 96426 },
  { event := event96487
    frameStart := 96426 },
  { event := event96488
    frameStart := 96426 },
  { event := event96489
    frameStart := 96426 },
  { event := event96490
    frameStart := 96426 },
  { event := event96491
    frameStart := 96426 },
  { event := event96492
    frameStart := 96426 },
  { event := event96493
    frameStart := 96426 },
  { event := event96494
    frameStart := 96426 },
  { event := event96495
    frameStart := 96426 }
]

def eventLeaf6031 : Array AnnotatedEvent := #[
  { event := event96496
    frameStart := 96426 },
  { event := event96497
    frameStart := 96426 },
  { event := event96498
    frameStart := 96426 },
  { event := event96499
    frameStart := 96426 },
  { event := event96500
    frameStart := 96426 },
  { event := event96501
    frameStart := 96426 },
  { event := event96502
    frameStart := 96426 },
  { event := event96503
    frameStart := 96426 },
  { event := event96504
    frameStart := 96426 },
  { event := event96505
    frameStart := 96426 },
  { event := event96506
    frameStart := 96426 },
  { event := event96507
    frameStart := 96426 },
  { event := event96508
    frameStart := 96426 },
  { event := event96509
    frameStart := 96426 },
  { event := event96510
    frameStart := 96426 },
  { event := event96511
    frameStart := 96426 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events376
