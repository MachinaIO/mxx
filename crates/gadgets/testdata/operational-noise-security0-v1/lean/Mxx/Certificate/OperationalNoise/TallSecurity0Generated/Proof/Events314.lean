import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events314

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact80384RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80384RawTermsValid :
    exact80384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80384 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22843⟩⟩) exact80384RawTerms .large 80216 (.finite 1811303510016) (some (80218))

def event80385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30119⟩⟩) 0 ⟨22843⟩ 80384

def event80386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30119⟩⟩) 1 ⟨30118⟩ 80206

def event80387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30119⟩⟩) (.sum [.predecessor 0 80385 .coefficient, .predecessor 1 80386 .coefficient])

def event80388 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30119⟩⟩, .operator (⟨80384, 0⟩, ⟨80206, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩, (1)⟩)

def event80389 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30119⟩⟩, .operator (⟨80384, 2⟩, ⟨80206, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨24792⟩⟩]⟩, (-1)⟩)

def event80390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30119⟩⟩) (.sum [.result 80384 .summary, .result 80206 .summary])

def exact80391RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80391RawTermsValid :
    exact80391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80391 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30119⟩⟩) exact80391RawTerms .large 80387 (.finite 1292539135285018636288) (some (80390))

def event80392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24727⟩⟩) 0 ⟨16872⟩ 3868

def event80393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24727⟩⟩) (.authority (.programFamilyFact))

def event80394 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24727⟩⟩) (.finite 3720)

def event80395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24729⟩⟩) 0 ⟨6689⟩ 5477

def event80396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24729⟩⟩) 1 ⟨24727⟩ 80394

def event80397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24729⟩⟩) (.authority (.operator))

def exact80398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩]

theorem exact80398RawTermsValid :
    exact80398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80398 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24729⟩⟩) exact80398RawTerms .large 80397 .exactZero (none)

def event80399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29819⟩⟩) 0 ⟨24729⟩ 80398

def event80400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29819⟩⟩) (.authority (.operator))

def exact80401RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩]

theorem exact80401RawTermsValid :
    exact80401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80401 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29819⟩⟩) exact80401RawTerms (.finite 8192) 80400 .exactZero (none)

def event80402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23373⟩⟩) 0 ⟨13156⟩ 3862

def event80403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23373⟩⟩) (.authority (.programFamilyFact))

def event80404 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23373⟩⟩) (.finite 3720)

def event80405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23374⟩⟩) 0 ⟨6689⟩ 5477

def event80406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23374⟩⟩) 1 ⟨23373⟩ 80404

def event80407 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23374⟩⟩) (.authority (.operator))

def exact80408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩]

theorem exact80408RawTermsValid :
    exact80408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80408 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23374⟩⟩) exact80408RawTerms .large 80407 .exactZero (none)

def event80409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25681⟩⟩) 0 ⟨23374⟩ 80408

def event80410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25681⟩⟩) (.authority (.operator))

def exact80411RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩]

theorem exact80411RawTermsValid :
    exact80411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25681⟩⟩) exact80411RawTerms (.finite 8192) 80410 .exactZero (none)

def event80412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13157⟩⟩) 0 ⟨13154⟩ 3851

def event80413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13157⟩⟩) 1 ⟨6567⟩ 79920

def event80414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13157⟩⟩) (.tensor (.predecessor 0 80412 .coefficient) (.predecessor 1 80413 .coefficient) true false)

def event80415 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13157⟩⟩, .operator (⟨3851, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80416RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80416RawTermsValid :
    exact80416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13157⟩⟩) exact80416RawTerms .large 80414 .exactZero (none)

def event80417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7245⟩⟩) 0 ⟨5539⟩ 79790

def event80418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7245⟩⟩) 1 ⟨6789⟩ 6973

def event80419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7245⟩⟩) (.product (.predecessor 0 80417 .coefficient) (.predecessor 1 80418 .coefficient) (⟨false, false, none, none, none⟩))

def event80420 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7245⟩⟩, .operator (⟨79790, 0⟩, ⟨6973, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def exact80421RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩]

theorem exact80421RawTermsValid :
    exact80421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80421 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7245⟩⟩) exact80421RawTerms .large 80419 .exactZero (none)

def event80422 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13158⟩⟩) 0 ⟨7245⟩ 80421

def event80423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13158⟩⟩) 1 ⟨13157⟩ 80416

def event80424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13158⟩⟩) (.sum [.predecessor 0 80422 .coefficient, .predecessor 1 80423 .coefficient])

def exact80425RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80425RawTermsValid :
    exact80425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13158⟩⟩) exact80425RawTerms .large 80424 .exactZero (none)

def event80426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13159⟩⟩) 0 ⟨13158⟩ 80425

def event80427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13159⟩⟩) 1 ⟨103⟩ 6965

def event80428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13159⟩⟩) (.sum [.predecessor 0 80426 .coefficient, .predecessor 1 80427 .coefficient])

def event80429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13159⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) [⟨.result 6965 .coefficient, false, none⟩])

def event80430 : Event := .survivorFold (1) 80429

def exact80431RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80431RawTermsValid :
    exact80431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80431 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13159⟩⟩) exact80431RawTerms .large 80428 (.finite 26) (some (80429))

def event80432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13160⟩⟩) 0 ⟨13159⟩ 80431

def event80433 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13160⟩⟩) 1 ⟨10240⟩ 3854

def event80434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13160⟩⟩) (.product (.predecessor 0 80432 .coefficient) (.predecessor 1 80433 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13160⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩) [⟨.result 3854 .coefficient, true, some 1⟩])

def event80436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13160⟩⟩) (.product (.result 80431 .summary) (.transfer 80435) (⟨false, false, none, none, none⟩))

def event80437 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13160⟩⟩, .operator (⟨80431, 1⟩, ⟨3854, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event80438 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13160⟩⟩, .operator (⟨80431, 0⟩, ⟨3854, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def exact80439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80439RawTermsValid :
    exact80439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13160⟩⟩) exact80439RawTerms .large 80434 (.finite 48256) (some (80436))

def event80440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10241⟩⟩) 0 ⟨10240⟩ 3854

def event80441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10241⟩⟩) 1 ⟨6567⟩ 79920

def event80442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10241⟩⟩) (.tensor (.predecessor 0 80440 .coefficient) (.predecessor 1 80441 .coefficient) true false)

def event80443 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10241⟩⟩, .operator (⟨3854, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80444RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80444RawTermsValid :
    exact80444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80444 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10241⟩⟩) exact80444RawTerms .large 80442 .exactZero (none)

def event80445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7225⟩⟩) 0 ⟨5539⟩ 79790

def event80446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7225⟩⟩) 1 ⟨6769⟩ 7014

def event80447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7225⟩⟩) (.product (.predecessor 0 80445 .coefficient) (.predecessor 1 80446 .coefficient) (⟨false, false, none, none, none⟩))

def event80448 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7225⟩⟩, .operator (⟨79790, 0⟩, ⟨7014, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩)

def exact80449RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩]

theorem exact80449RawTermsValid :
    exact80449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80449 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7225⟩⟩) exact80449RawTerms .large 80447 .exactZero (none)

def event80450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10242⟩⟩) 0 ⟨7225⟩ 80449

def event80451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10242⟩⟩) 1 ⟨10241⟩ 80444

def event80452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10242⟩⟩) (.sum [.predecessor 0 80450 .coefficient, .predecessor 1 80451 .coefficient])

def exact80453RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80453RawTermsValid :
    exact80453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10242⟩⟩) exact80453RawTerms .large 80452 .exactZero (none)

def event80454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10243⟩⟩) 0 ⟨10242⟩ 80453

def event80455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10243⟩⟩) 1 ⟨83⟩ 7006

def event80456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10243⟩⟩) (.sum [.predecessor 0 80454 .coefficient, .predecessor 1 80455 .coefficient])

def event80457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10243⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩) [⟨.result 7006 .coefficient, false, none⟩])

def event80458 : Event := .survivorFold (1) 80457

def exact80459RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80459RawTermsValid :
    exact80459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80459 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10243⟩⟩) exact80459RawTerms .large 80456 (.finite 26) (some (80457))

def event80460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10244⟩⟩) 0 ⟨10243⟩ 80459

def event80461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10244⟩⟩) 1 ⟨7880⟩ 7003

def event80462 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10244⟩⟩) (.product (.predecessor 0 80460 .coefficient) (.predecessor 1 80461 .coefficient) (⟨false, false, none, none, none⟩))

def event80463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10244⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) [⟨.result 6999 .coefficient, false, none⟩])

def event80464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10244⟩⟩) (.product (.result 80459 .summary) (.transfer 80463) (⟨false, false, none, none, none⟩))

def event80465 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10244⟩⟩, .operator (⟨80459, 1⟩, ⟨7003, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (-1)⟩)

def event80466 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨10244⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973)

def event80467 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10244⟩⟩, .relation 80466 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (-1)⟩)

def event80468 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10244⟩⟩, .operator (⟨80459, 0⟩, ⟨7003, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩)

def exact80469RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (-1)⟩]

theorem exact80469RawTermsValid :
    exact80469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10244⟩⟩) exact80469RawTerms .large 80462 (.finite 95420416) (some (80464))

def event80470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13161⟩⟩) 0 ⟨10244⟩ 80469

def event80471 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13161⟩⟩) 1 ⟨13160⟩ 80439

def event80472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13161⟩⟩) (.sum [.predecessor 0 80470 .coefficient, .predecessor 1 80471 .coefficient])

def event80473 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13161⟩⟩, .operator (⟨80469, 1⟩, ⟨80439, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def event80474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13161⟩⟩) (.sum [.result 80469 .summary, .result 80439 .summary])

def exact80475RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80475RawTermsValid :
    exact80475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80475 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13161⟩⟩) exact80475RawTerms .large 80472 (.finite 95468672) (some (80474))

def event80476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25682⟩⟩) 0 ⟨13161⟩ 80475

def event80477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25682⟩⟩) 1 ⟨25681⟩ 80411

def event80478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25682⟩⟩) (.product (.predecessor 0 80476 .coefficient) (.predecessor 1 80477 .coefficient) (⟨false, false, none, none, none⟩))

def event80479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25682⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) [⟨.result 80411 .coefficient, false, none⟩])

def event80480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25682⟩⟩) (.product (.result 80475 .summary) (.transfer 80479) (⟨false, false, none, none, none⟩))

def event80481 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25682⟩⟩, .operator (⟨80475, 1⟩, ⟨80411, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩)

def event80482 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25682⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25681⟩⟩) ⟨23374⟩ 80408)

def event80483 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25682⟩⟩, .relation 80482 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (-1)⟩)

def event80484 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25682⟩⟩, .operator (⟨80475, 0⟩, ⟨80411, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩)

def exact80485RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (-1)⟩]

theorem exact80485RawTermsValid :
    exact80485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25682⟩⟩) exact80485RawTerms .large 80478 (.finite 350371553738752) (some (80480))

def event80486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20176⟩⟩) 0 ⟨13156⟩ 3862

def event80487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20176⟩⟩) (.authority (.relationPreimageSource ⟨25⟩))

def exact80488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩]

theorem exact80488RawTermsValid :
    exact80488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20176⟩⟩) exact80488RawTerms (.finite 136065468) 80487 .exactZero (none)

def event80489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20178⟩⟩) 0 ⟨20176⟩ 80488

def event80490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20178⟩⟩) 1 ⟨2348⟩ 4

def event80491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20178⟩⟩) (.scale (.predecessor 0 80489 .coefficient) (.value (.predecessor 1 80490 .coefficient)))

def exact80492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩]

theorem exact80492RawTermsValid :
    exact80492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80492 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20178⟩⟩) exact80492RawTerms (.finite 136065468) 80491 .exactZero (none)

def event80493 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20179⟩⟩) 0 ⟨5541⟩ 80012

def event80494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20179⟩⟩) 1 ⟨20178⟩ 80492

def event80495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20179⟩⟩) (.product (.predecessor 0 80493 .coefficient) (.predecessor 1 80494 .coefficient) (⟨false, false, none, none, none⟩))

def event80496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20179⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) [⟨.result 80488 .coefficient, false, none⟩])

def event80497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20179⟩⟩) (.product (.result 80012 .summary) (.transfer 80496) (⟨false, false, none, none, none⟩))

def event80498 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20179⟩⟩, .operator (⟨80012, 0⟩, ⟨80492, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩)

def event80499 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20177⟩⟩)

def event80500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80501 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80503 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80507 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80507

def event80509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80505

def event80510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80508 .coefficient) (.value (.predecessor 1 80509 .coefficient)))

def event80511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80511

def event80513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80503

def event80514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80512 .coefficient, .predecessor 1 80513 .coefficient])

def event80515 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80515

def event80517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80501

def event80518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80517 .coefficient))

def event80519 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13154⟩⟩) 0 ⟨5536⟩ 80519

def event80521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13154⟩⟩) (.authority (.programFamilyFact))

def exact80522RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80522RawTermsValid :
    exact80522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13154⟩⟩) exact80522RawTerms (.finite 58) 80521 .exactZero (none)

def event80523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10240⟩⟩) 0 ⟨5536⟩ 80519

def event80524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10240⟩⟩) (.authority (.programFamilyFact))

def exact80525RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩, (1)⟩]

theorem exact80525RawTermsValid :
    exact80525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10240⟩⟩) exact80525RawTerms (.finite 58) 80524 .exactZero (none)

def event80526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 0 ⟨10240⟩ 80525

def event80527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 1 ⟨13154⟩ 80522

def event80528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.product (.predecessor 0 80526 .coefficient) (.predecessor 1 80527 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩) [⟨.result 80525 .coefficient, true, some 1⟩, ⟨.result 80522 .coefficient, true, some 1⟩])

def event80530 : Event := .survivorFold (1) 80529

def exact80531RawTerms : List Term := []

theorem exact80531RawTermsValid :
    exact80531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13155⟩⟩) exact80531RawTerms (.finite 3364) 80528 (.finite 3364) (some (80529))

def event80532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13156⟩⟩) 0 ⟨13155⟩ 80531

def event80533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.identity (.predecessor 0 80532 .coefficient))

def event80534 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.finite 3364)

def event80535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20176⟩⟩) 0 ⟨13156⟩ 80534

def event80536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20176⟩⟩) (.authority (.relationPreimageSource ⟨25⟩))

def exact80537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩]

theorem exact80537RawTermsValid :
    exact80537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20176⟩⟩) exact80537RawTerms (.finite 136065468) 80536 .exactZero (none)

def event80538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact80539RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact80539RawTermsValid :
    exact80539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80539 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact80539RawTerms .large 80538 .exactZero (none)

def event80540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20177⟩⟩) 0 ⟨6⟩ 80539

def event80541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20177⟩⟩) 1 ⟨20176⟩ 80537

def event80542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20177⟩⟩) (.product (.predecessor 0 80540 .coefficient) (.predecessor 1 80541 .coefficient) (⟨false, false, none, none, none⟩))

def event80543 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20177⟩⟩, .operator (⟨80539, 0⟩, ⟨80537, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩)

def exact80544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩]

theorem exact80544RawTermsValid :
    exact80544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20177⟩⟩) exact80544RawTerms .large 80542 .exactZero (none)

def event80545 : Event := .preFoldPolynomial 80544 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩] .exactZero none

def exact80546RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩, (1)⟩]

def event80546 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20177⟩⟩) 80545 exact80546RawTerms .large 80542 .exactZero (none)

def event80547 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25685⟩⟩)

def event80548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80549 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80553 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80555 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80555

def event80557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80553

def event80558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80556 .coefficient) (.value (.predecessor 1 80557 .coefficient)))

def event80559 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80559

def event80561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80551

def event80562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80560 .coefficient, .predecessor 1 80561 .coefficient])

def event80563 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80563

def event80565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80549

def event80566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80565 .coefficient))

def event80567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13154⟩⟩) 0 ⟨5536⟩ 80567

def event80569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13154⟩⟩) (.authority (.programFamilyFact))

def exact80570RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80570RawTermsValid :
    exact80570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80570 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13154⟩⟩) exact80570RawTerms (.finite 58) 80569 .exactZero (none)

def event80571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10240⟩⟩) 0 ⟨5536⟩ 80567

def event80572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10240⟩⟩) (.authority (.programFamilyFact))

def exact80573RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩, (1)⟩]

theorem exact80573RawTermsValid :
    exact80573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10240⟩⟩) exact80573RawTerms (.finite 58) 80572 .exactZero (none)

def event80574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 0 ⟨10240⟩ 80573

def event80575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 1 ⟨13154⟩ 80570

def event80576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.product (.predecessor 0 80574 .coefficient) (.predecessor 1 80575 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80577 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13155⟩⟩, .operator (⟨80573, 0⟩, ⟨80570, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩)

def exact80578RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80578RawTermsValid :
    exact80578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80578 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13155⟩⟩) exact80578RawTerms (.finite 3364) 80576 .exactZero (none)

def event80579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13156⟩⟩) 0 ⟨13155⟩ 80578

def event80580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.identity (.predecessor 0 80579 .coefficient))

def event80581 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.finite 3364)

def event80582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23373⟩⟩) 0 ⟨13156⟩ 80581

def event80583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23373⟩⟩) (.authority (.programFamilyFact))

def event80584 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23373⟩⟩) (.finite 3720)

def event80585 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event80586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23374⟩⟩) 0 ⟨6689⟩ 80585

def event80587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23374⟩⟩) 1 ⟨23373⟩ 80584

def event80588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23374⟩⟩) (.authority (.operator))

def exact80589RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩]

theorem exact80589RawTermsValid :
    exact80589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80589 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23374⟩⟩) exact80589RawTerms .large 80588 .exactZero (none)

def event80590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25681⟩⟩) 0 ⟨23374⟩ 80589

def event80591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25681⟩⟩) (.authority (.operator))

def exact80592RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩]

theorem exact80592RawTermsValid :
    exact80592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25681⟩⟩) exact80592RawTerms (.finite 8192) 80591 .exactZero (none)

def event80593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event80594 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event80595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13250⟩⟩) 0 ⟨13156⟩ 80581

def event80596 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13250⟩⟩) 1 ⟨110⟩ 80594

def event80597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13250⟩⟩) (.sum [.predecessor 0 80595 .coefficient, .predecessor 1 80596 .coefficient])

def event80598 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13250⟩⟩) (.finite 3364)

def event80599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13251⟩⟩) 0 ⟨13250⟩ 80598

def event80600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13251⟩⟩) (.identity (.predecessor 0 80599 .coefficient))

def exact80601RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80601RawTermsValid :
    exact80601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80601 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13251⟩⟩) exact80601RawTerms (.finite 3364) 80600 .exactZero (none)

def event80602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact80603RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80603RawTermsValid :
    exact80603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact80603RawTerms .large 80602 .exactZero (none)

def event80604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13252⟩⟩) 0 ⟨6544⟩ 80603

def event80605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13252⟩⟩) 1 ⟨13251⟩ 80601

def event80606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13252⟩⟩) (.product (.predecessor 0 80604 .coefficient) (.predecessor 1 80605 .coefficient) (⟨false, false, none, none, none⟩))

def event80607 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13252⟩⟩, .operator (⟨80603, 0⟩, ⟨80601, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80608RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80608RawTermsValid :
    exact80608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80608 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13252⟩⟩) exact80608RawTerms .large 80606 .exactZero (none)

def event80609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 80585

def event80610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact80611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact80611RawTermsValid :
    exact80611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80611 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact80611RawTerms .large 80610 .exactZero (none)

def event80612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6789⟩⟩) 0 ⟨6757⟩ 80611

def event80613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6789⟩⟩) (.identity (.predecessor 0 80612 .coefficient))

def exact80614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩]

theorem exact80614RawTermsValid :
    exact80614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6789⟩⟩) exact80614RawTerms .large 80613 .exactZero (none)

def event80615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7879⟩⟩) 0 ⟨6789⟩ 80614

def event80616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7879⟩⟩) (.authority (.operator))

def exact80617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩]

theorem exact80617RawTermsValid :
    exact80617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7879⟩⟩) exact80617RawTerms (.finite 8192) 80616 .exactZero (none)

def event80618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7880⟩⟩) 0 ⟨7879⟩ 80617

def event80619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7880⟩⟩) 1 ⟨2348⟩ 80551

def event80620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7880⟩⟩) (.scale (.predecessor 0 80618 .coefficient) (.value (.predecessor 1 80619 .coefficient)))

def exact80621RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩]

theorem exact80621RawTermsValid :
    exact80621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7880⟩⟩) exact80621RawTerms (.finite 8192) 80620 .exactZero (none)

def event80622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6769⟩⟩) 0 ⟨6757⟩ 80611

def event80623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6769⟩⟩) (.identity (.predecessor 0 80622 .coefficient))

def exact80624RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩]

theorem exact80624RawTermsValid :
    exact80624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6769⟩⟩) exact80624RawTerms .large 80623 .exactZero (none)

def event80625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7881⟩⟩) 0 ⟨6769⟩ 80624

def event80626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7881⟩⟩) 1 ⟨7880⟩ 80621

def event80627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7881⟩⟩) (.product (.predecessor 0 80625 .coefficient) (.predecessor 1 80626 .coefficient) (⟨false, false, none, none, none⟩))

def event80628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7881⟩⟩, .operator (⟨80624, 0⟩, ⟨80621, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩)

def exact80629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩]

theorem exact80629RawTermsValid :
    exact80629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7881⟩⟩) exact80629RawTerms .large 80627 .exactZero (none)

def event80630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13253⟩⟩) 0 ⟨7881⟩ 80629

def event80631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13253⟩⟩) 1 ⟨13252⟩ 80608

def event80632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13253⟩⟩) (.sum [.predecessor 0 80630 .coefficient, .predecessor 1 80631 .coefficient])

def exact80633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80633RawTermsValid :
    exact80633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13253⟩⟩) exact80633RawTerms .large 80632 .exactZero (none)

def event80634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25684⟩⟩) 0 ⟨13253⟩ 80633

def event80635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25684⟩⟩) 1 ⟨25681⟩ 80592

def event80636 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25684⟩⟩) (.product (.predecessor 0 80634 .coefficient) (.predecessor 1 80635 .coefficient) (⟨false, false, none, none, none⟩))

def event80637 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25684⟩⟩, .operator (⟨80633, 0⟩, ⟨80592, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩)

def event80638 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25684⟩⟩, .operator (⟨80633, 1⟩, ⟨80592, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩)

def event80639 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25684⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25681⟩⟩) ⟨23374⟩ 80589)

def eventLeaf5024 : Array AnnotatedEvent := #[
  { event := event80384
    frameStart := 0 },
  { event := event80385
    frameStart := 0 },
  { event := event80386
    frameStart := 0 },
  { event := event80387
    frameStart := 0 },
  { event := event80388
    frameStart := 0 },
  { event := event80389
    frameStart := 0 },
  { event := event80390
    frameStart := 0 },
  { event := event80391
    frameStart := 0 },
  { event := event80392
    frameStart := 0 },
  { event := event80393
    frameStart := 0 },
  { event := event80394
    frameStart := 0 },
  { event := event80395
    frameStart := 0 },
  { event := event80396
    frameStart := 0 },
  { event := event80397
    frameStart := 0 },
  { event := event80398
    frameStart := 0 },
  { event := event80399
    frameStart := 0 }
]

def eventLeaf5025 : Array AnnotatedEvent := #[
  { event := event80400
    frameStart := 0 },
  { event := event80401
    frameStart := 0 },
  { event := event80402
    frameStart := 0 },
  { event := event80403
    frameStart := 0 },
  { event := event80404
    frameStart := 0 },
  { event := event80405
    frameStart := 0 },
  { event := event80406
    frameStart := 0 },
  { event := event80407
    frameStart := 0 },
  { event := event80408
    frameStart := 0 },
  { event := event80409
    frameStart := 0 },
  { event := event80410
    frameStart := 0 },
  { event := event80411
    frameStart := 0 },
  { event := event80412
    frameStart := 0 },
  { event := event80413
    frameStart := 0 },
  { event := event80414
    frameStart := 0 },
  { event := event80415
    frameStart := 0 }
]

def eventLeaf5026 : Array AnnotatedEvent := #[
  { event := event80416
    frameStart := 0 },
  { event := event80417
    frameStart := 0 },
  { event := event80418
    frameStart := 0 },
  { event := event80419
    frameStart := 0 },
  { event := event80420
    frameStart := 0 },
  { event := event80421
    frameStart := 0 },
  { event := event80422
    frameStart := 0 },
  { event := event80423
    frameStart := 0 },
  { event := event80424
    frameStart := 0 },
  { event := event80425
    frameStart := 0 },
  { event := event80426
    frameStart := 0 },
  { event := event80427
    frameStart := 0 },
  { event := event80428
    frameStart := 0 },
  { event := event80429
    frameStart := 0 },
  { event := event80430
    frameStart := 0 },
  { event := event80431
    frameStart := 0 }
]

def eventLeaf5027 : Array AnnotatedEvent := #[
  { event := event80432
    frameStart := 0 },
  { event := event80433
    frameStart := 0 },
  { event := event80434
    frameStart := 0 },
  { event := event80435
    frameStart := 0 },
  { event := event80436
    frameStart := 0 },
  { event := event80437
    frameStart := 0 },
  { event := event80438
    frameStart := 0 },
  { event := event80439
    frameStart := 0 },
  { event := event80440
    frameStart := 0 },
  { event := event80441
    frameStart := 0 },
  { event := event80442
    frameStart := 0 },
  { event := event80443
    frameStart := 0 },
  { event := event80444
    frameStart := 0 },
  { event := event80445
    frameStart := 0 },
  { event := event80446
    frameStart := 0 },
  { event := event80447
    frameStart := 0 }
]

def eventLeaf5028 : Array AnnotatedEvent := #[
  { event := event80448
    frameStart := 0 },
  { event := event80449
    frameStart := 0 },
  { event := event80450
    frameStart := 0 },
  { event := event80451
    frameStart := 0 },
  { event := event80452
    frameStart := 0 },
  { event := event80453
    frameStart := 0 },
  { event := event80454
    frameStart := 0 },
  { event := event80455
    frameStart := 0 },
  { event := event80456
    frameStart := 0 },
  { event := event80457
    frameStart := 0 },
  { event := event80458
    frameStart := 0 },
  { event := event80459
    frameStart := 0 },
  { event := event80460
    frameStart := 0 },
  { event := event80461
    frameStart := 0 },
  { event := event80462
    frameStart := 0 },
  { event := event80463
    frameStart := 0 }
]

def eventLeaf5029 : Array AnnotatedEvent := #[
  { event := event80464
    frameStart := 0 },
  { event := event80465
    frameStart := 0 },
  { event := event80466
    frameStart := 0 },
  { event := event80467
    frameStart := 0 },
  { event := event80468
    frameStart := 0 },
  { event := event80469
    frameStart := 0 },
  { event := event80470
    frameStart := 0 },
  { event := event80471
    frameStart := 0 },
  { event := event80472
    frameStart := 0 },
  { event := event80473
    frameStart := 0 },
  { event := event80474
    frameStart := 0 },
  { event := event80475
    frameStart := 0 },
  { event := event80476
    frameStart := 0 },
  { event := event80477
    frameStart := 0 },
  { event := event80478
    frameStart := 0 },
  { event := event80479
    frameStart := 0 }
]

def eventLeaf5030 : Array AnnotatedEvent := #[
  { event := event80480
    frameStart := 0 },
  { event := event80481
    frameStart := 0 },
  { event := event80482
    frameStart := 0 },
  { event := event80483
    frameStart := 0 },
  { event := event80484
    frameStart := 0 },
  { event := event80485
    frameStart := 0 },
  { event := event80486
    frameStart := 0 },
  { event := event80487
    frameStart := 0 },
  { event := event80488
    frameStart := 0 },
  { event := event80489
    frameStart := 0 },
  { event := event80490
    frameStart := 0 },
  { event := event80491
    frameStart := 0 },
  { event := event80492
    frameStart := 0 },
  { event := event80493
    frameStart := 0 },
  { event := event80494
    frameStart := 0 },
  { event := event80495
    frameStart := 0 }
]

def eventLeaf5031 : Array AnnotatedEvent := #[
  { event := event80496
    frameStart := 0 },
  { event := event80497
    frameStart := 0 },
  { event := event80498
    frameStart := 0 },
  { event := event80499
    frameStart := 80499 },
  { event := event80500
    frameStart := 80499 },
  { event := event80501
    frameStart := 80499 },
  { event := event80502
    frameStart := 80499 },
  { event := event80503
    frameStart := 80499 },
  { event := event80504
    frameStart := 80499 },
  { event := event80505
    frameStart := 80499 },
  { event := event80506
    frameStart := 80499 },
  { event := event80507
    frameStart := 80499 },
  { event := event80508
    frameStart := 80499 },
  { event := event80509
    frameStart := 80499 },
  { event := event80510
    frameStart := 80499 },
  { event := event80511
    frameStart := 80499 }
]

def eventLeaf5032 : Array AnnotatedEvent := #[
  { event := event80512
    frameStart := 80499 },
  { event := event80513
    frameStart := 80499 },
  { event := event80514
    frameStart := 80499 },
  { event := event80515
    frameStart := 80499 },
  { event := event80516
    frameStart := 80499 },
  { event := event80517
    frameStart := 80499 },
  { event := event80518
    frameStart := 80499 },
  { event := event80519
    frameStart := 80499 },
  { event := event80520
    frameStart := 80499 },
  { event := event80521
    frameStart := 80499 },
  { event := event80522
    frameStart := 80499 },
  { event := event80523
    frameStart := 80499 },
  { event := event80524
    frameStart := 80499 },
  { event := event80525
    frameStart := 80499 },
  { event := event80526
    frameStart := 80499 },
  { event := event80527
    frameStart := 80499 }
]

def eventLeaf5033 : Array AnnotatedEvent := #[
  { event := event80528
    frameStart := 80499 },
  { event := event80529
    frameStart := 80499 },
  { event := event80530
    frameStart := 80499 },
  { event := event80531
    frameStart := 80499 },
  { event := event80532
    frameStart := 80499 },
  { event := event80533
    frameStart := 80499 },
  { event := event80534
    frameStart := 80499 },
  { event := event80535
    frameStart := 80499 },
  { event := event80536
    frameStart := 80499 },
  { event := event80537
    frameStart := 80499 },
  { event := event80538
    frameStart := 80499 },
  { event := event80539
    frameStart := 80499 },
  { event := event80540
    frameStart := 80499 },
  { event := event80541
    frameStart := 80499 },
  { event := event80542
    frameStart := 80499 },
  { event := event80543
    frameStart := 80499 }
]

def eventLeaf5034 : Array AnnotatedEvent := #[
  { event := event80544
    frameStart := 80499 },
  { event := event80545
    frameStart := 80499 },
  { event := event80546
    frameStart := 80499 },
  { event := event80547
    frameStart := 80547 },
  { event := event80548
    frameStart := 80547 },
  { event := event80549
    frameStart := 80547 },
  { event := event80550
    frameStart := 80547 },
  { event := event80551
    frameStart := 80547 },
  { event := event80552
    frameStart := 80547 },
  { event := event80553
    frameStart := 80547 },
  { event := event80554
    frameStart := 80547 },
  { event := event80555
    frameStart := 80547 },
  { event := event80556
    frameStart := 80547 },
  { event := event80557
    frameStart := 80547 },
  { event := event80558
    frameStart := 80547 },
  { event := event80559
    frameStart := 80547 }
]

def eventLeaf5035 : Array AnnotatedEvent := #[
  { event := event80560
    frameStart := 80547 },
  { event := event80561
    frameStart := 80547 },
  { event := event80562
    frameStart := 80547 },
  { event := event80563
    frameStart := 80547 },
  { event := event80564
    frameStart := 80547 },
  { event := event80565
    frameStart := 80547 },
  { event := event80566
    frameStart := 80547 },
  { event := event80567
    frameStart := 80547 },
  { event := event80568
    frameStart := 80547 },
  { event := event80569
    frameStart := 80547 },
  { event := event80570
    frameStart := 80547 },
  { event := event80571
    frameStart := 80547 },
  { event := event80572
    frameStart := 80547 },
  { event := event80573
    frameStart := 80547 },
  { event := event80574
    frameStart := 80547 },
  { event := event80575
    frameStart := 80547 }
]

def eventLeaf5036 : Array AnnotatedEvent := #[
  { event := event80576
    frameStart := 80547 },
  { event := event80577
    frameStart := 80547 },
  { event := event80578
    frameStart := 80547 },
  { event := event80579
    frameStart := 80547 },
  { event := event80580
    frameStart := 80547 },
  { event := event80581
    frameStart := 80547 },
  { event := event80582
    frameStart := 80547 },
  { event := event80583
    frameStart := 80547 },
  { event := event80584
    frameStart := 80547 },
  { event := event80585
    frameStart := 80547 },
  { event := event80586
    frameStart := 80547 },
  { event := event80587
    frameStart := 80547 },
  { event := event80588
    frameStart := 80547 },
  { event := event80589
    frameStart := 80547 },
  { event := event80590
    frameStart := 80547 },
  { event := event80591
    frameStart := 80547 }
]

def eventLeaf5037 : Array AnnotatedEvent := #[
  { event := event80592
    frameStart := 80547 },
  { event := event80593
    frameStart := 80547 },
  { event := event80594
    frameStart := 80547 },
  { event := event80595
    frameStart := 80547 },
  { event := event80596
    frameStart := 80547 },
  { event := event80597
    frameStart := 80547 },
  { event := event80598
    frameStart := 80547 },
  { event := event80599
    frameStart := 80547 },
  { event := event80600
    frameStart := 80547 },
  { event := event80601
    frameStart := 80547 },
  { event := event80602
    frameStart := 80547 },
  { event := event80603
    frameStart := 80547 },
  { event := event80604
    frameStart := 80547 },
  { event := event80605
    frameStart := 80547 },
  { event := event80606
    frameStart := 80547 },
  { event := event80607
    frameStart := 80547 }
]

def eventLeaf5038 : Array AnnotatedEvent := #[
  { event := event80608
    frameStart := 80547 },
  { event := event80609
    frameStart := 80547 },
  { event := event80610
    frameStart := 80547 },
  { event := event80611
    frameStart := 80547 },
  { event := event80612
    frameStart := 80547 },
  { event := event80613
    frameStart := 80547 },
  { event := event80614
    frameStart := 80547 },
  { event := event80615
    frameStart := 80547 },
  { event := event80616
    frameStart := 80547 },
  { event := event80617
    frameStart := 80547 },
  { event := event80618
    frameStart := 80547 },
  { event := event80619
    frameStart := 80547 },
  { event := event80620
    frameStart := 80547 },
  { event := event80621
    frameStart := 80547 },
  { event := event80622
    frameStart := 80547 },
  { event := event80623
    frameStart := 80547 }
]

def eventLeaf5039 : Array AnnotatedEvent := #[
  { event := event80624
    frameStart := 80547 },
  { event := event80625
    frameStart := 80547 },
  { event := event80626
    frameStart := 80547 },
  { event := event80627
    frameStart := 80547 },
  { event := event80628
    frameStart := 80547 },
  { event := event80629
    frameStart := 80547 },
  { event := event80630
    frameStart := 80547 },
  { event := event80631
    frameStart := 80547 },
  { event := event80632
    frameStart := 80547 },
  { event := event80633
    frameStart := 80547 },
  { event := event80634
    frameStart := 80547 },
  { event := event80635
    frameStart := 80547 },
  { event := event80636
    frameStart := 80547 },
  { event := event80637
    frameStart := 80547 },
  { event := event80638
    frameStart := 80547 },
  { event := event80639
    frameStart := 80547 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events314
