import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events329

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact84224RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84224RawTermsValid :
    exact84224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21691⟩⟩) exact84224RawTerms .large 84056 (.finite 1811303510016) (some (84058))

def event84225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28303⟩⟩) 0 ⟨21691⟩ 84224

def event84226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28303⟩⟩) 1 ⟨28302⟩ 84046

def event84227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28303⟩⟩) (.sum [.predecessor 0 84225 .coefficient, .predecessor 1 84226 .coefficient])

def event84228 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28303⟩⟩, .operator (⟨84224, 0⟩, ⟨84046, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩)

def event84229 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28303⟩⟩, .operator (⟨84224, 2⟩, ⟨84046, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (-1)⟩)

def event84230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28303⟩⟩) (.sum [.result 84224 .summary, .result 84046 .summary])

def exact84231RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84231RawTermsValid :
    exact84231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84231 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28303⟩⟩) exact84231RawTerms .large 84227 (.finite 1292180536164689260544) (some (84230))

def event84232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24223⟩⟩) 0 ⟨16060⟩ 4052

def event84233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24223⟩⟩) (.authority (.programFamilyFact))

def event84234 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24223⟩⟩) (.finite 3720)

def event84235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24225⟩⟩) 0 ⟨6689⟩ 5477

def event84236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24225⟩⟩) 1 ⟨24223⟩ 84234

def event84237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24225⟩⟩) (.authority (.operator))

def exact84238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24225⟩⟩]⟩, (1)⟩]

theorem exact84238RawTermsValid :
    exact84238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84238 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24225⟩⟩) exact84238RawTerms .large 84237 .exactZero (none)

def event84239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28083⟩⟩) 0 ⟨24225⟩ 84238

def event84240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28083⟩⟩) (.authority (.operator))

def exact84241RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28083⟩⟩]⟩, (1)⟩]

theorem exact84241RawTermsValid :
    exact84241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84241 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28083⟩⟩) exact84241RawTerms (.finite 8192) 84240 .exactZero (none)

def event84242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23625⟩⟩) 0 ⟨14426⟩ 4046

def event84243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23625⟩⟩) (.authority (.programFamilyFact))

def event84244 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23625⟩⟩) (.finite 3720)

def event84245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23626⟩⟩) 0 ⟨6689⟩ 5477

def event84246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23626⟩⟩) 1 ⟨23625⟩ 84244

def event84247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23626⟩⟩) (.authority (.operator))

def exact84248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩, (1)⟩]

theorem exact84248RawTermsValid :
    exact84248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23626⟩⟩) exact84248RawTerms .large 84247 .exactZero (none)

def event84249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26143⟩⟩) 0 ⟨23626⟩ 84248

def event84250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26143⟩⟩) (.authority (.operator))

def exact84251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (1)⟩]

theorem exact84251RawTermsValid :
    exact84251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84251 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26143⟩⟩) exact84251RawTerms (.finite 8192) 84250 .exactZero (none)

def event84252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11554⟩⟩) 0 ⟨11553⟩ 4035

def event84253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11554⟩⟩) 1 ⟨6567⟩ 79920

def event84254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11554⟩⟩) (.tensor (.predecessor 0 84252 .coefficient) (.predecessor 1 84253 .coefficient) true false)

def event84255 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11554⟩⟩, .operator (⟨4035, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84256RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84256RawTermsValid :
    exact84256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11554⟩⟩) exact84256RawTerms .large 84254 .exactZero (none)

def event84257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7236⟩⟩) 0 ⟨5539⟩ 79790

def event84258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7236⟩⟩) 1 ⟨6780⟩ 10981

def event84259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7236⟩⟩) (.product (.predecessor 0 84257 .coefficient) (.predecessor 1 84258 .coefficient) (⟨false, false, none, none, none⟩))

def event84260 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7236⟩⟩, .operator (⟨79790, 0⟩, ⟨10981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact84261RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact84261RawTermsValid :
    exact84261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84261 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7236⟩⟩) exact84261RawTerms .large 84259 .exactZero (none)

def event84262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11555⟩⟩) 0 ⟨7236⟩ 84261

def event84263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11555⟩⟩) 1 ⟨11554⟩ 84256

def event84264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11555⟩⟩) (.sum [.predecessor 0 84262 .coefficient, .predecessor 1 84263 .coefficient])

def exact84265RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84265RawTermsValid :
    exact84265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84265 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11555⟩⟩) exact84265RawTerms .large 84264 .exactZero (none)

def event84266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11556⟩⟩) 0 ⟨11555⟩ 84265

def event84267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11556⟩⟩) 1 ⟨94⟩ 10973

def event84268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11556⟩⟩) (.sum [.predecessor 0 84266 .coefficient, .predecessor 1 84267 .coefficient])

def event84269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11556⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) [⟨.result 10973 .coefficient, false, none⟩])

def event84270 : Event := .survivorFold (1) 84269

def exact84271RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84271RawTermsValid :
    exact84271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11556⟩⟩) exact84271RawTerms .large 84268 (.finite 26) (some (84269))

def event84272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14427⟩⟩) 0 ⟨11556⟩ 84271

def event84273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14427⟩⟩) 1 ⟨14424⟩ 4038

def event84274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14427⟩⟩) (.product (.predecessor 0 84272 .coefficient) (.predecessor 1 84273 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14427⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩) [⟨.result 4038 .coefficient, true, some 1⟩])

def event84276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14427⟩⟩) (.product (.result 84271 .summary) (.transfer 84275) (⟨false, false, none, none, none⟩))

def event84277 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14427⟩⟩, .operator (⟨84271, 1⟩, ⟨4038, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event84278 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14427⟩⟩, .operator (⟨84271, 0⟩, ⟨4038, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact84279RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact84279RawTermsValid :
    exact84279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84279 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14427⟩⟩) exact84279RawTerms .large 84274 (.finite 18304) (some (84276))

def event84280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14428⟩⟩) 0 ⟨14424⟩ 4038

def event84281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14428⟩⟩) 1 ⟨6567⟩ 79920

def event84282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14428⟩⟩) (.tensor (.predecessor 0 84280 .coefficient) (.predecessor 1 84281 .coefficient) true false)

def event84283 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14428⟩⟩, .operator (⟨4038, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84284RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84284RawTermsValid :
    exact84284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84284 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14428⟩⟩) exact84284RawTerms .large 84282 .exactZero (none)

def event84285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7217⟩⟩) 0 ⟨5539⟩ 79790

def event84286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7217⟩⟩) 1 ⟨6761⟩ 11022

def event84287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7217⟩⟩) (.product (.predecessor 0 84285 .coefficient) (.predecessor 1 84286 .coefficient) (⟨false, false, none, none, none⟩))

def event84288 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7217⟩⟩, .operator (⟨79790, 0⟩, ⟨11022, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩)

def exact84289RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩]

theorem exact84289RawTermsValid :
    exact84289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84289 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7217⟩⟩) exact84289RawTerms .large 84287 .exactZero (none)

def event84290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14429⟩⟩) 0 ⟨7217⟩ 84289

def event84291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14429⟩⟩) 1 ⟨14428⟩ 84284

def event84292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14429⟩⟩) (.sum [.predecessor 0 84290 .coefficient, .predecessor 1 84291 .coefficient])

def exact84293RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84293RawTermsValid :
    exact84293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84293 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14429⟩⟩) exact84293RawTerms .large 84292 .exactZero (none)

def event84294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14430⟩⟩) 0 ⟨14429⟩ 84293

def event84295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14430⟩⟩) 1 ⟨75⟩ 11014

def event84296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14430⟩⟩) (.sum [.predecessor 0 84294 .coefficient, .predecessor 1 84295 .coefficient])

def event84297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14430⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩) [⟨.result 11014 .coefficient, false, none⟩])

def event84298 : Event := .survivorFold (1) 84297

def exact84299RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84299RawTermsValid :
    exact84299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84299 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14430⟩⟩) exact84299RawTerms .large 84296 (.finite 26) (some (84297))

def event84300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14431⟩⟩) 0 ⟨14430⟩ 84299

def event84301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14431⟩⟩) 1 ⟨7856⟩ 11011

def event84302 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14431⟩⟩) (.product (.predecessor 0 84300 .coefficient) (.predecessor 1 84301 .coefficient) (⟨false, false, none, none, none⟩))

def event84303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14431⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) [⟨.result 11007 .coefficient, false, none⟩])

def event84304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14431⟩⟩) (.product (.result 84299 .summary) (.transfer 84303) (⟨false, false, none, none, none⟩))

def event84305 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14431⟩⟩, .operator (⟨84299, 1⟩, ⟨11011, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (-1)⟩)

def event84306 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14431⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7855⟩⟩) ⟨6780⟩ 10981)

def event84307 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14431⟩⟩, .relation 84306 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (-1)⟩)

def event84308 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14431⟩⟩, .operator (⟨84299, 0⟩, ⟨11011, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩)

def exact84309RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (-1)⟩]

theorem exact84309RawTermsValid :
    exact84309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84309 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14431⟩⟩) exact84309RawTerms .large 84302 (.finite 95420416) (some (84304))

def event84310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14432⟩⟩) 0 ⟨14431⟩ 84309

def event84311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14432⟩⟩) 1 ⟨14427⟩ 84279

def event84312 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14432⟩⟩) (.sum [.predecessor 0 84310 .coefficient, .predecessor 1 84311 .coefficient])

def event84313 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14432⟩⟩, .operator (⟨84309, 1⟩, ⟨84279, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def event84314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14432⟩⟩) (.sum [.result 84309 .summary, .result 84279 .summary])

def exact84315RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84315RawTermsValid :
    exact84315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84315 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14432⟩⟩) exact84315RawTerms .large 84312 (.finite 95438720) (some (84314))

def event84316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26144⟩⟩) 0 ⟨14432⟩ 84315

def event84317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26144⟩⟩) 1 ⟨26143⟩ 84251

def event84318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26144⟩⟩) (.product (.predecessor 0 84316 .coefficient) (.predecessor 1 84317 .coefficient) (⟨false, false, none, none, none⟩))

def event84319 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26144⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) [⟨.result 84251 .coefficient, false, none⟩])

def event84320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26144⟩⟩) (.product (.result 84315 .summary) (.transfer 84319) (⟨false, false, none, none, none⟩))

def event84321 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26144⟩⟩, .operator (⟨84315, 1⟩, ⟨84251, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (-1)⟩)

def event84322 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26144⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26143⟩⟩) ⟨23626⟩ 84248)

def event84323 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26144⟩⟩, .relation 84322 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩, (-1)⟩)

def event84324 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26144⟩⟩, .operator (⟨84315, 0⟩, ⟨84251, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (1)⟩)

def exact84325RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩, (-1)⟩]

theorem exact84325RawTermsValid :
    exact84325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26144⟩⟩) exact84325RawTerms .large 84318 (.finite 350261629419520) (some (84320))

def event84326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19600⟩⟩) 0 ⟨14426⟩ 4046

def event84327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19600⟩⟩) (.authority (.relationPreimageSource ⟨16⟩))

def exact84328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩]

theorem exact84328RawTermsValid :
    exact84328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84328 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19600⟩⟩) exact84328RawTerms (.finite 136065468) 84327 .exactZero (none)

def event84329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19602⟩⟩) 0 ⟨19600⟩ 84328

def event84330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19602⟩⟩) 1 ⟨2348⟩ 4

def event84331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19602⟩⟩) (.scale (.predecessor 0 84329 .coefficient) (.value (.predecessor 1 84330 .coefficient)))

def exact84332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩]

theorem exact84332RawTermsValid :
    exact84332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19602⟩⟩) exact84332RawTerms (.finite 136065468) 84331 .exactZero (none)

def event84333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19603⟩⟩) 0 ⟨5541⟩ 80012

def event84334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19603⟩⟩) 1 ⟨19602⟩ 84332

def event84335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19603⟩⟩) (.product (.predecessor 0 84333 .coefficient) (.predecessor 1 84334 .coefficient) (⟨false, false, none, none, none⟩))

def event84336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19603⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩) [⟨.result 84328 .coefficient, false, none⟩])

def event84337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19603⟩⟩) (.product (.result 80012 .summary) (.transfer 84336) (⟨false, false, none, none, none⟩))

def event84338 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19603⟩⟩, .operator (⟨80012, 0⟩, ⟨84332, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩)

def event84339 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19601⟩⟩)

def event84340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84341 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84343 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84345 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84347 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84347

def event84349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84345

def event84350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84348 .coefficient) (.value (.predecessor 1 84349 .coefficient)))

def event84351 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84351

def event84353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84343

def event84354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84352 .coefficient, .predecessor 1 84353 .coefficient])

def event84355 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84355

def event84357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84341

def event84358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84357 .coefficient))

def event84359 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11553⟩⟩) 0 ⟨5536⟩ 84359

def event84361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11553⟩⟩) (.authority (.programFamilyFact))

def exact84362RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩], []⟩, (1)⟩]

theorem exact84362RawTermsValid :
    exact84362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84362 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11553⟩⟩) exact84362RawTerms (.finite 22) 84361 .exactZero (none)

def event84363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14424⟩⟩) 0 ⟨5536⟩ 84359

def event84364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14424⟩⟩) (.authority (.programFamilyFact))

def exact84365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩]

theorem exact84365RawTermsValid :
    exact84365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14424⟩⟩) exact84365RawTerms (.finite 22) 84364 .exactZero (none)

def event84366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 0 ⟨14424⟩ 84365

def event84367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 1 ⟨11553⟩ 84362

def event84368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14425⟩⟩) (.product (.predecessor 0 84366 .coefficient) (.predecessor 1 84367 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14425⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩) [⟨.result 84365 .coefficient, true, some 1⟩, ⟨.result 84362 .coefficient, true, some 1⟩])

def event84370 : Event := .survivorFold (1) 84369

def exact84371RawTerms : List Term := []

theorem exact84371RawTermsValid :
    exact84371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84371 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14425⟩⟩) exact84371RawTerms (.finite 484) 84368 (.finite 484) (some (84369))

def event84372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14426⟩⟩) 0 ⟨14425⟩ 84371

def event84373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.identity (.predecessor 0 84372 .coefficient))

def event84374 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.finite 484)

def event84375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19600⟩⟩) 0 ⟨14426⟩ 84374

def event84376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19600⟩⟩) (.authority (.relationPreimageSource ⟨16⟩))

def exact84377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩]

theorem exact84377RawTermsValid :
    exact84377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19600⟩⟩) exact84377RawTerms (.finite 136065468) 84376 .exactZero (none)

def event84378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact84379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact84379RawTermsValid :
    exact84379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact84379RawTerms .large 84378 .exactZero (none)

def event84380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19601⟩⟩) 0 ⟨6⟩ 84379

def event84381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19601⟩⟩) 1 ⟨19600⟩ 84377

def event84382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19601⟩⟩) (.product (.predecessor 0 84380 .coefficient) (.predecessor 1 84381 .coefficient) (⟨false, false, none, none, none⟩))

def event84383 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19601⟩⟩, .operator (⟨84379, 0⟩, ⟨84377, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩)

def exact84384RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩]

theorem exact84384RawTermsValid :
    exact84384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84384 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19601⟩⟩) exact84384RawTerms .large 84382 .exactZero (none)

def event84385 : Event := .preFoldPolynomial 84384 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩] .exactZero none

def exact84386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19600⟩⟩]⟩, (1)⟩]

def event84386 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19601⟩⟩) 84385 exact84386RawTerms .large 84382 .exactZero (none)

def event84387 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26147⟩⟩)

def event84388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84391 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84393 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84395 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84395

def event84397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84393

def event84398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84396 .coefficient) (.value (.predecessor 1 84397 .coefficient)))

def event84399 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84399

def event84401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84391

def event84402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84400 .coefficient, .predecessor 1 84401 .coefficient])

def event84403 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84403

def event84405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84389

def event84406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84405 .coefficient))

def event84407 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11553⟩⟩) 0 ⟨5536⟩ 84407

def event84409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11553⟩⟩) (.authority (.programFamilyFact))

def exact84410RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩], []⟩, (1)⟩]

theorem exact84410RawTermsValid :
    exact84410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11553⟩⟩) exact84410RawTerms (.finite 22) 84409 .exactZero (none)

def event84411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14424⟩⟩) 0 ⟨5536⟩ 84407

def event84412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14424⟩⟩) (.authority (.programFamilyFact))

def exact84413RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩]

theorem exact84413RawTermsValid :
    exact84413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14424⟩⟩) exact84413RawTerms (.finite 22) 84412 .exactZero (none)

def event84414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 0 ⟨14424⟩ 84413

def event84415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 1 ⟨11553⟩ 84410

def event84416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14425⟩⟩) (.product (.predecessor 0 84414 .coefficient) (.predecessor 1 84415 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84417 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14425⟩⟩, .operator (⟨84413, 0⟩, ⟨84410, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩)

def exact84418RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩]

theorem exact84418RawTermsValid :
    exact84418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14425⟩⟩) exact84418RawTerms (.finite 484) 84416 .exactZero (none)

def event84419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14426⟩⟩) 0 ⟨14425⟩ 84418

def event84420 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.identity (.predecessor 0 84419 .coefficient))

def event84421 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.finite 484)

def event84422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23625⟩⟩) 0 ⟨14426⟩ 84421

def event84423 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23625⟩⟩) (.authority (.programFamilyFact))

def event84424 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23625⟩⟩) (.finite 3720)

def event84425 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event84426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23626⟩⟩) 0 ⟨6689⟩ 84425

def event84427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23626⟩⟩) 1 ⟨23625⟩ 84424

def event84428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23626⟩⟩) (.authority (.operator))

def exact84429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23626⟩⟩]⟩, (1)⟩]

theorem exact84429RawTermsValid :
    exact84429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84429 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23626⟩⟩) exact84429RawTerms .large 84428 .exactZero (none)

def event84430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26143⟩⟩) 0 ⟨23626⟩ 84429

def event84431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26143⟩⟩) (.authority (.operator))

def exact84432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (1)⟩]

theorem exact84432RawTermsValid :
    exact84432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84432 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26143⟩⟩) exact84432RawTerms (.finite 8192) 84431 .exactZero (none)

def event84433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event84434 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event84435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14531⟩⟩) 0 ⟨14426⟩ 84421

def event84436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14531⟩⟩) 1 ⟨110⟩ 84434

def event84437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14531⟩⟩) (.sum [.predecessor 0 84435 .coefficient, .predecessor 1 84436 .coefficient])

def event84438 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14531⟩⟩) (.finite 484)

def event84439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14532⟩⟩) 0 ⟨14531⟩ 84438

def event84440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14532⟩⟩) (.identity (.predecessor 0 84439 .coefficient))

def exact84441RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩]

theorem exact84441RawTermsValid :
    exact84441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84441 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14532⟩⟩) exact84441RawTerms (.finite 484) 84440 .exactZero (none)

def event84442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact84443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84443RawTermsValid :
    exact84443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84443 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact84443RawTerms .large 84442 .exactZero (none)

def event84444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14533⟩⟩) 0 ⟨6544⟩ 84443

def event84445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14533⟩⟩) 1 ⟨14532⟩ 84441

def event84446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14533⟩⟩) (.product (.predecessor 0 84444 .coefficient) (.predecessor 1 84445 .coefficient) (⟨false, false, none, none, none⟩))

def event84447 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14533⟩⟩, .operator (⟨84443, 0⟩, ⟨84441, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84448RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84448RawTermsValid :
    exact84448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84448 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14533⟩⟩) exact84448RawTerms .large 84446 .exactZero (none)

def event84449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 84425

def event84450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact84451RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact84451RawTermsValid :
    exact84451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84451 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact84451RawTerms .large 84450 .exactZero (none)

def event84452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6780⟩⟩) 0 ⟨6757⟩ 84451

def event84453 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6780⟩⟩) (.identity (.predecessor 0 84452 .coefficient))

def exact84454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact84454RawTermsValid :
    exact84454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84454 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6780⟩⟩) exact84454RawTerms .large 84453 .exactZero (none)

def event84455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7855⟩⟩) 0 ⟨6780⟩ 84454

def event84456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7855⟩⟩) (.authority (.operator))

def exact84457RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact84457RawTermsValid :
    exact84457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7855⟩⟩) exact84457RawTerms (.finite 8192) 84456 .exactZero (none)

def event84458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 0 ⟨7855⟩ 84457

def event84459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 1 ⟨2348⟩ 84391

def event84460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7856⟩⟩) (.scale (.predecessor 0 84458 .coefficient) (.value (.predecessor 1 84459 .coefficient)))

def exact84461RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact84461RawTermsValid :
    exact84461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7856⟩⟩) exact84461RawTerms (.finite 8192) 84460 .exactZero (none)

def event84462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6761⟩⟩) 0 ⟨6757⟩ 84451

def event84463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6761⟩⟩) (.identity (.predecessor 0 84462 .coefficient))

def exact84464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩]

theorem exact84464RawTermsValid :
    exact84464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6761⟩⟩) exact84464RawTerms .large 84463 .exactZero (none)

def event84465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 0 ⟨6761⟩ 84464

def event84466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 1 ⟨7856⟩ 84461

def event84467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7857⟩⟩) (.product (.predecessor 0 84465 .coefficient) (.predecessor 1 84466 .coefficient) (⟨false, false, none, none, none⟩))

def event84468 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7857⟩⟩, .operator (⟨84464, 0⟩, ⟨84461, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩)

def exact84469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact84469RawTermsValid :
    exact84469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7857⟩⟩) exact84469RawTerms .large 84467 .exactZero (none)

def event84470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14534⟩⟩) 0 ⟨7857⟩ 84469

def event84471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14534⟩⟩) 1 ⟨14533⟩ 84448

def event84472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14534⟩⟩) (.sum [.predecessor 0 84470 .coefficient, .predecessor 1 84471 .coefficient])

def exact84473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84473RawTermsValid :
    exact84473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84473 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14534⟩⟩) exact84473RawTerms .large 84472 .exactZero (none)

def event84474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26146⟩⟩) 0 ⟨14534⟩ 84473

def event84475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26146⟩⟩) 1 ⟨26143⟩ 84432

def event84476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26146⟩⟩) (.product (.predecessor 0 84474 .coefficient) (.predecessor 1 84475 .coefficient) (⟨false, false, none, none, none⟩))

def event84477 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26146⟩⟩, .operator (⟨84473, 0⟩, ⟨84432, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (1)⟩)

def event84478 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26146⟩⟩, .operator (⟨84473, 1⟩, ⟨84432, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩, (-1)⟩)

def event84479 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26146⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26143⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26143⟩⟩) ⟨23626⟩ 84429)

def eventLeaf5264 : Array AnnotatedEvent := #[
  { event := event84224
    frameStart := 0 },
  { event := event84225
    frameStart := 0 },
  { event := event84226
    frameStart := 0 },
  { event := event84227
    frameStart := 0 },
  { event := event84228
    frameStart := 0 },
  { event := event84229
    frameStart := 0 },
  { event := event84230
    frameStart := 0 },
  { event := event84231
    frameStart := 0 },
  { event := event84232
    frameStart := 0 },
  { event := event84233
    frameStart := 0 },
  { event := event84234
    frameStart := 0 },
  { event := event84235
    frameStart := 0 },
  { event := event84236
    frameStart := 0 },
  { event := event84237
    frameStart := 0 },
  { event := event84238
    frameStart := 0 },
  { event := event84239
    frameStart := 0 }
]

def eventLeaf5265 : Array AnnotatedEvent := #[
  { event := event84240
    frameStart := 0 },
  { event := event84241
    frameStart := 0 },
  { event := event84242
    frameStart := 0 },
  { event := event84243
    frameStart := 0 },
  { event := event84244
    frameStart := 0 },
  { event := event84245
    frameStart := 0 },
  { event := event84246
    frameStart := 0 },
  { event := event84247
    frameStart := 0 },
  { event := event84248
    frameStart := 0 },
  { event := event84249
    frameStart := 0 },
  { event := event84250
    frameStart := 0 },
  { event := event84251
    frameStart := 0 },
  { event := event84252
    frameStart := 0 },
  { event := event84253
    frameStart := 0 },
  { event := event84254
    frameStart := 0 },
  { event := event84255
    frameStart := 0 }
]

def eventLeaf5266 : Array AnnotatedEvent := #[
  { event := event84256
    frameStart := 0 },
  { event := event84257
    frameStart := 0 },
  { event := event84258
    frameStart := 0 },
  { event := event84259
    frameStart := 0 },
  { event := event84260
    frameStart := 0 },
  { event := event84261
    frameStart := 0 },
  { event := event84262
    frameStart := 0 },
  { event := event84263
    frameStart := 0 },
  { event := event84264
    frameStart := 0 },
  { event := event84265
    frameStart := 0 },
  { event := event84266
    frameStart := 0 },
  { event := event84267
    frameStart := 0 },
  { event := event84268
    frameStart := 0 },
  { event := event84269
    frameStart := 0 },
  { event := event84270
    frameStart := 0 },
  { event := event84271
    frameStart := 0 }
]

def eventLeaf5267 : Array AnnotatedEvent := #[
  { event := event84272
    frameStart := 0 },
  { event := event84273
    frameStart := 0 },
  { event := event84274
    frameStart := 0 },
  { event := event84275
    frameStart := 0 },
  { event := event84276
    frameStart := 0 },
  { event := event84277
    frameStart := 0 },
  { event := event84278
    frameStart := 0 },
  { event := event84279
    frameStart := 0 },
  { event := event84280
    frameStart := 0 },
  { event := event84281
    frameStart := 0 },
  { event := event84282
    frameStart := 0 },
  { event := event84283
    frameStart := 0 },
  { event := event84284
    frameStart := 0 },
  { event := event84285
    frameStart := 0 },
  { event := event84286
    frameStart := 0 },
  { event := event84287
    frameStart := 0 }
]

def eventLeaf5268 : Array AnnotatedEvent := #[
  { event := event84288
    frameStart := 0 },
  { event := event84289
    frameStart := 0 },
  { event := event84290
    frameStart := 0 },
  { event := event84291
    frameStart := 0 },
  { event := event84292
    frameStart := 0 },
  { event := event84293
    frameStart := 0 },
  { event := event84294
    frameStart := 0 },
  { event := event84295
    frameStart := 0 },
  { event := event84296
    frameStart := 0 },
  { event := event84297
    frameStart := 0 },
  { event := event84298
    frameStart := 0 },
  { event := event84299
    frameStart := 0 },
  { event := event84300
    frameStart := 0 },
  { event := event84301
    frameStart := 0 },
  { event := event84302
    frameStart := 0 },
  { event := event84303
    frameStart := 0 }
]

def eventLeaf5269 : Array AnnotatedEvent := #[
  { event := event84304
    frameStart := 0 },
  { event := event84305
    frameStart := 0 },
  { event := event84306
    frameStart := 0 },
  { event := event84307
    frameStart := 0 },
  { event := event84308
    frameStart := 0 },
  { event := event84309
    frameStart := 0 },
  { event := event84310
    frameStart := 0 },
  { event := event84311
    frameStart := 0 },
  { event := event84312
    frameStart := 0 },
  { event := event84313
    frameStart := 0 },
  { event := event84314
    frameStart := 0 },
  { event := event84315
    frameStart := 0 },
  { event := event84316
    frameStart := 0 },
  { event := event84317
    frameStart := 0 },
  { event := event84318
    frameStart := 0 },
  { event := event84319
    frameStart := 0 }
]

def eventLeaf5270 : Array AnnotatedEvent := #[
  { event := event84320
    frameStart := 0 },
  { event := event84321
    frameStart := 0 },
  { event := event84322
    frameStart := 0 },
  { event := event84323
    frameStart := 0 },
  { event := event84324
    frameStart := 0 },
  { event := event84325
    frameStart := 0 },
  { event := event84326
    frameStart := 0 },
  { event := event84327
    frameStart := 0 },
  { event := event84328
    frameStart := 0 },
  { event := event84329
    frameStart := 0 },
  { event := event84330
    frameStart := 0 },
  { event := event84331
    frameStart := 0 },
  { event := event84332
    frameStart := 0 },
  { event := event84333
    frameStart := 0 },
  { event := event84334
    frameStart := 0 },
  { event := event84335
    frameStart := 0 }
]

def eventLeaf5271 : Array AnnotatedEvent := #[
  { event := event84336
    frameStart := 0 },
  { event := event84337
    frameStart := 0 },
  { event := event84338
    frameStart := 0 },
  { event := event84339
    frameStart := 84339 },
  { event := event84340
    frameStart := 84339 },
  { event := event84341
    frameStart := 84339 },
  { event := event84342
    frameStart := 84339 },
  { event := event84343
    frameStart := 84339 },
  { event := event84344
    frameStart := 84339 },
  { event := event84345
    frameStart := 84339 },
  { event := event84346
    frameStart := 84339 },
  { event := event84347
    frameStart := 84339 },
  { event := event84348
    frameStart := 84339 },
  { event := event84349
    frameStart := 84339 },
  { event := event84350
    frameStart := 84339 },
  { event := event84351
    frameStart := 84339 }
]

def eventLeaf5272 : Array AnnotatedEvent := #[
  { event := event84352
    frameStart := 84339 },
  { event := event84353
    frameStart := 84339 },
  { event := event84354
    frameStart := 84339 },
  { event := event84355
    frameStart := 84339 },
  { event := event84356
    frameStart := 84339 },
  { event := event84357
    frameStart := 84339 },
  { event := event84358
    frameStart := 84339 },
  { event := event84359
    frameStart := 84339 },
  { event := event84360
    frameStart := 84339 },
  { event := event84361
    frameStart := 84339 },
  { event := event84362
    frameStart := 84339 },
  { event := event84363
    frameStart := 84339 },
  { event := event84364
    frameStart := 84339 },
  { event := event84365
    frameStart := 84339 },
  { event := event84366
    frameStart := 84339 },
  { event := event84367
    frameStart := 84339 }
]

def eventLeaf5273 : Array AnnotatedEvent := #[
  { event := event84368
    frameStart := 84339 },
  { event := event84369
    frameStart := 84339 },
  { event := event84370
    frameStart := 84339 },
  { event := event84371
    frameStart := 84339 },
  { event := event84372
    frameStart := 84339 },
  { event := event84373
    frameStart := 84339 },
  { event := event84374
    frameStart := 84339 },
  { event := event84375
    frameStart := 84339 },
  { event := event84376
    frameStart := 84339 },
  { event := event84377
    frameStart := 84339 },
  { event := event84378
    frameStart := 84339 },
  { event := event84379
    frameStart := 84339 },
  { event := event84380
    frameStart := 84339 },
  { event := event84381
    frameStart := 84339 },
  { event := event84382
    frameStart := 84339 },
  { event := event84383
    frameStart := 84339 }
]

def eventLeaf5274 : Array AnnotatedEvent := #[
  { event := event84384
    frameStart := 84339 },
  { event := event84385
    frameStart := 84339 },
  { event := event84386
    frameStart := 84339 },
  { event := event84387
    frameStart := 84387 },
  { event := event84388
    frameStart := 84387 },
  { event := event84389
    frameStart := 84387 },
  { event := event84390
    frameStart := 84387 },
  { event := event84391
    frameStart := 84387 },
  { event := event84392
    frameStart := 84387 },
  { event := event84393
    frameStart := 84387 },
  { event := event84394
    frameStart := 84387 },
  { event := event84395
    frameStart := 84387 },
  { event := event84396
    frameStart := 84387 },
  { event := event84397
    frameStart := 84387 },
  { event := event84398
    frameStart := 84387 },
  { event := event84399
    frameStart := 84387 }
]

def eventLeaf5275 : Array AnnotatedEvent := #[
  { event := event84400
    frameStart := 84387 },
  { event := event84401
    frameStart := 84387 },
  { event := event84402
    frameStart := 84387 },
  { event := event84403
    frameStart := 84387 },
  { event := event84404
    frameStart := 84387 },
  { event := event84405
    frameStart := 84387 },
  { event := event84406
    frameStart := 84387 },
  { event := event84407
    frameStart := 84387 },
  { event := event84408
    frameStart := 84387 },
  { event := event84409
    frameStart := 84387 },
  { event := event84410
    frameStart := 84387 },
  { event := event84411
    frameStart := 84387 },
  { event := event84412
    frameStart := 84387 },
  { event := event84413
    frameStart := 84387 },
  { event := event84414
    frameStart := 84387 },
  { event := event84415
    frameStart := 84387 }
]

def eventLeaf5276 : Array AnnotatedEvent := #[
  { event := event84416
    frameStart := 84387 },
  { event := event84417
    frameStart := 84387 },
  { event := event84418
    frameStart := 84387 },
  { event := event84419
    frameStart := 84387 },
  { event := event84420
    frameStart := 84387 },
  { event := event84421
    frameStart := 84387 },
  { event := event84422
    frameStart := 84387 },
  { event := event84423
    frameStart := 84387 },
  { event := event84424
    frameStart := 84387 },
  { event := event84425
    frameStart := 84387 },
  { event := event84426
    frameStart := 84387 },
  { event := event84427
    frameStart := 84387 },
  { event := event84428
    frameStart := 84387 },
  { event := event84429
    frameStart := 84387 },
  { event := event84430
    frameStart := 84387 },
  { event := event84431
    frameStart := 84387 }
]

def eventLeaf5277 : Array AnnotatedEvent := #[
  { event := event84432
    frameStart := 84387 },
  { event := event84433
    frameStart := 84387 },
  { event := event84434
    frameStart := 84387 },
  { event := event84435
    frameStart := 84387 },
  { event := event84436
    frameStart := 84387 },
  { event := event84437
    frameStart := 84387 },
  { event := event84438
    frameStart := 84387 },
  { event := event84439
    frameStart := 84387 },
  { event := event84440
    frameStart := 84387 },
  { event := event84441
    frameStart := 84387 },
  { event := event84442
    frameStart := 84387 },
  { event := event84443
    frameStart := 84387 },
  { event := event84444
    frameStart := 84387 },
  { event := event84445
    frameStart := 84387 },
  { event := event84446
    frameStart := 84387 },
  { event := event84447
    frameStart := 84387 }
]

def eventLeaf5278 : Array AnnotatedEvent := #[
  { event := event84448
    frameStart := 84387 },
  { event := event84449
    frameStart := 84387 },
  { event := event84450
    frameStart := 84387 },
  { event := event84451
    frameStart := 84387 },
  { event := event84452
    frameStart := 84387 },
  { event := event84453
    frameStart := 84387 },
  { event := event84454
    frameStart := 84387 },
  { event := event84455
    frameStart := 84387 },
  { event := event84456
    frameStart := 84387 },
  { event := event84457
    frameStart := 84387 },
  { event := event84458
    frameStart := 84387 },
  { event := event84459
    frameStart := 84387 },
  { event := event84460
    frameStart := 84387 },
  { event := event84461
    frameStart := 84387 },
  { event := event84462
    frameStart := 84387 },
  { event := event84463
    frameStart := 84387 }
]

def eventLeaf5279 : Array AnnotatedEvent := #[
  { event := event84464
    frameStart := 84387 },
  { event := event84465
    frameStart := 84387 },
  { event := event84466
    frameStart := 84387 },
  { event := event84467
    frameStart := 84387 },
  { event := event84468
    frameStart := 84387 },
  { event := event84469
    frameStart := 84387 },
  { event := event84470
    frameStart := 84387 },
  { event := event84471
    frameStart := 84387 },
  { event := event84472
    frameStart := 84387 },
  { event := event84473
    frameStart := 84387 },
  { event := event84474
    frameStart := 84387 },
  { event := event84475
    frameStart := 84387 },
  { event := event84476
    frameStart := 84387 },
  { event := event84477
    frameStart := 84387 },
  { event := event84478
    frameStart := 84387 },
  { event := event84479
    frameStart := 84387 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events329
