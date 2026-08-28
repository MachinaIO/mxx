import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events295

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact75520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩]⟩, (1)⟩]

theorem exact75520RawTermsValid :
    exact75520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7404⟩⟩) exact75520RawTerms .large 75518 .exactZero (none)

def event75521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10793⟩⟩) 0 ⟨7404⟩ 75520

def event75522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10793⟩⟩) 1 ⟨10752⟩ 61278

def event75523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10793⟩⟩) (.sum [.predecessor 0 75521 .coefficient, .predecessor 1 75522 .coefficient])

def exact75524RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75524RawTermsValid :
    exact75524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10793⟩⟩) exact75524RawTerms .large 75523 .exactZero (none)

def event75525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10794⟩⟩) 0 ⟨10793⟩ 75524

def event75526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10794⟩⟩) 1 ⟨26⟩ 75515

def event75527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10794⟩⟩) (.sum [.predecessor 0 75525 .coefficient, .predecessor 1 75526 .coefficient])

def event75528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10794⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨26⟩⟩]⟩) [⟨.result 75515 .coefficient, false, none⟩])

def event75529 : Event := .survivorFold (1) 75528

def exact75530RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75530RawTermsValid :
    exact75530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10794⟩⟩) exact75530RawTerms .large 75527 (.finite 26) (some (75528))

def event75531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10795⟩⟩) 0 ⟨10794⟩ 75530

def event75532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10795⟩⟩) 1 ⟨9584⟩ 15984

def event75533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10795⟩⟩) (.product (.predecessor 0 75531 .coefficient) (.predecessor 1 75532 .coefficient) (⟨false, false, none, none, none⟩))

def event75534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10795⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event75535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10795⟩⟩) (.product (.result 75530 .summary) (.transfer 75534) (⟨false, false, none, none, none⟩))

def event75536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .operator (⟨75530, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event75537 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨10795⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event75538 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 18, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event75539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 17, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event75540 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 16, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event75541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 15, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event75542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 14, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event75543 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 13, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event75544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 12, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event75545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 11, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event75546 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 10, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event75547 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 9, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩)

def event75548 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 8, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event75549 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 7, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩)

def event75550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 6, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event75551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 5, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event75552 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 4, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def event75553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩)

def event75554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event75555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩)

def event75556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .relation 75537 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event75557 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10795⟩⟩, .operator (⟨75530, 0⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩)

def exact75558RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩]

theorem exact75558RawTermsValid :
    exact75558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75558 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10795⟩⟩) exact75558RawTerms .large 75533 (.finite 279172874240) (some (75535))

def event75559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71476⟩⟩) 0 ⟨10795⟩ 75558

def event75560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71476⟩⟩) 1 ⟨71475⟩ 75513

def event75561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71476⟩⟩) (.sum [.predecessor 0 75559 .coefficient, .predecessor 1 75560 .coefficient])

def event75562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 19⟩, ⟨75513, 37⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event75563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 18⟩, ⟨75513, 36⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event75564 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 17⟩, ⟨75513, 35⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event75565 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 16⟩, ⟨75513, 34⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event75566 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 15⟩, ⟨75513, 33⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event75567 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 14⟩, ⟨75513, 32⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event75568 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 13⟩, ⟨75513, 31⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event75569 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 12⟩, ⟨75513, 30⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def event75570 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 11⟩, ⟨75513, 29⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩)

def event75571 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 10⟩, ⟨75513, 28⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩)

def event75572 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 9⟩, ⟨75513, 27⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩)

def event75573 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 8⟩, ⟨75513, 26⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩)

def event75574 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 7⟩, ⟨75513, 25⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩)

def event75575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 6⟩, ⟨75513, 24⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩)

def event75576 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 5⟩, ⟨75513, 23⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩)

def event75577 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 4⟩, ⟨75513, 22⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩)

def event75578 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 3⟩, ⟨75513, 21⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event75579 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 2⟩, ⟨75513, 20⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩)

def event75580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71476⟩⟩, .operator (⟨75558, 1⟩, ⟨75513, 19⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event75581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71476⟩⟩) (.sum [.result 75558 .summary, .result 75513 .summary])

def exact75582RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75582RawTermsValid :
    exact75582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75582 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71476⟩⟩) exact75582RawTerms .large 75561 (.finite 66805187227601152574551644069558752530002375679672372) (some (75581))

def event75583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71477⟩⟩) 0 ⟨71476⟩ 75582

def event75584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71477⟩⟩) 1 ⟨9498⟩ 16104

def event75585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71477⟩⟩) (.product (.predecessor 0 75583 .coefficient) (.predecessor 1 75584 .coefficient) (⟨false, false, none, none, none⟩))

def event75586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71477⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) [⟨.result 16100 .coefficient, false, none⟩])

def event75587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71477⟩⟩) (.product (.result 75582 .summary) (.transfer 75586) (⟨false, false, none, none, none⟩))

def event75588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 6⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (1)⟩)

def event75589 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75590 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75589 0, ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 8⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75592 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75593 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75592 0, ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75594 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 9⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75595 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75596 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75595 0, ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75597 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 10⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75598 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75599 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75598 0, ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 12⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75601 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75602 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75601 0, ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75603 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 13⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75604 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75605 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75604 0, ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75606 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 14⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75607 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75608 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75607 0, ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75609 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 16⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75610 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75611 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75610 0, ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75612 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 17⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75613 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75613 0, ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75615 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 19⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75616 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75616 0, ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 1⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75619 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75620 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75619 0, ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 2⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75622 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75622 0, ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 3⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75625 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75625 0, ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 4⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75628 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75628 0, ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75630 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 5⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75631 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75632 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75631 0, ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75633 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 7⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75634 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75634 0, ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 11⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75637 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75637 0, ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 15⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75640 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75641 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75640 0, ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75642 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 18⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (-1)⟩)

def event75643 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71477⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9497⟩⟩) ⟨7241⟩ 16097)

def event75644 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .relation 75643 0, ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71477⟩⟩, .operator (⟨75582, 0⟩, ⟨16104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (1)⟩)

def exact75646RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩]

theorem exact75646RawTermsValid :
    exact75646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71477⟩⟩) exact75646RawTerms .large 75585 (.finite 717315235864259647099013782854467978167293655866246524336865280) (some (75587))

def event75647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71478⟩⟩) 0 ⟨71477⟩ 75646

def event75648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71478⟩⟩) 1 ⟨67612⟩ 61243

def event75649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71478⟩⟩) (.sum [.predecessor 0 75647 .coefficient, .predecessor 1 75648 .coefficient])

def event75650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 6⟩, ⟨61243, 24⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (-1)⟩)

def event75651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 8⟩, ⟨61243, 26⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75652 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 9⟩, ⟨61243, 27⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75653 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 10⟩, ⟨61243, 28⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 12⟩, ⟨61243, 30⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 13⟩, ⟨61243, 31⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75656 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 14⟩, ⟨61243, 32⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75657 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 16⟩, ⟨61243, 34⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75658 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 17⟩, ⟨61243, 35⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75659 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 19⟩, ⟨61243, 37⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 1⟩, ⟨61243, 19⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 2⟩, ⟨61243, 20⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 3⟩, ⟨61243, 21⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 4⟩, ⟨61243, 22⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 5⟩, ⟨61243, 23⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 7⟩, ⟨61243, 25⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 11⟩, ⟨61243, 29⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75667 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 15⟩, ⟨61243, 33⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75668 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71478⟩⟩, .operator (⟨75646, 18⟩, ⟨61243, 36⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨7241⟩⟩]⟩, (1)⟩)

def event75669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71478⟩⟩) (.sum [.result 75646 .summary, .result 61243 .summary])

def exact75670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75670RawTermsValid :
    exact75670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71478⟩⟩) exact75670RawTerms .large 75649 (.finite 717315235864259647099013782854474880280923984914290088855535616) (some (75669))

def event75671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71479⟩⟩) 0 ⟨71478⟩ 75670

def event75672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71479⟩⟩) 1 ⟨7142⟩ 16094

def event75673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71479⟩⟩) (.product (.predecessor 0 75671 .coefficient) (.predecessor 1 75672 .coefficient) (⟨false, false, none, none, none⟩))

def event75674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71479⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) [⟨.result 16090 .coefficient, false, none⟩])

def event75675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71479⟩⟩) (.product (.result 75670 .summary) (.transfer 75674) (⟨false, false, none, none, none⟩))

def event75676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 6⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (1)⟩)

def event75677 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75677 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event75679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 8⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75680 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75680 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 9⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75683 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75684 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75683 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 10⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75686 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75686 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 12⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75689 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75689 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75691 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 13⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75692 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75692 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 14⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75695 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75695 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 16⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75698 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75698 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 17⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75701 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75701 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 19⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75704 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75704 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 1⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75707 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75707 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75709 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 2⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75710 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75710 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 3⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75713 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75713 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75715 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 4⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75716 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75717 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75716 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 5⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75719 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75719 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 7⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75722 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75722 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 11⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75725 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75725 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75727 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 15⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75728 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75729 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75728 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 18⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (-1)⟩)

def event75731 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71479⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7141⟩⟩) ⟨7036⟩ 16087)

def event75732 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .relation 75731 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71479⟩⟩, .operator (⟨75670, 0⟩, ⟨16094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (1)⟩)

def exact75734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75734RawTermsValid :
    exact75734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71479⟩⟩) exact75734RawTerms .large 75673 (.finite 7702113697398803698856913678033037845150209519672183236728648848208035840) (some (75675))

def event75735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25⟩⟩) (.authority (.operator))

def exact75736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨25⟩⟩]⟩, (1)⟩]

theorem exact75736RawTermsValid :
    exact75736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25⟩⟩) exact75736RawTerms (.finite 26) 75735 .exactZero (none)

def event75737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5394⟩⟩) (.authority (.operator))

def event75738 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5394⟩⟩) (.finite 655345)

def event75739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10376⟩⟩) 0 ⟨10325⟩ 3083

def event75740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10376⟩⟩) 1 ⟨5394⟩ 75738

def event75741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10376⟩⟩) (.sum [.predecessor 0 75739 .coefficient, .predecessor 1 75740 .coefficient])

def event75742 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10376⟩⟩) (.finite 1310705)

def event75743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10435⟩⟩) 0 ⟨10376⟩ 75742

def event75744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10435⟩⟩) 1 ⟨5426⟩ 38

def event75745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10435⟩⟩) (.identity (.predecessor 1 75744 .coefficient))

def event75746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10435⟩⟩) (.finite 655360)

def event75747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10436⟩⟩) 0 ⟨10435⟩ 75746

def event75748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10436⟩⟩) 1 ⟨2370⟩ 4

def event75749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10436⟩⟩) (.sum [.predecessor 0 75747 .coefficient, .predecessor 1 75748 .coefficient])

def event75750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10436⟩⟩) (.finite 655361)

def event75751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10437⟩⟩) 0 ⟨0⟩ 20

def event75752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10437⟩⟩) 1 ⟨10435⟩ 75746

def event75753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10437⟩⟩) 2 ⟨10436⟩ 75750

def event75754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10437⟩⟩) 3 ⟨136⟩ 6

def event75755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10437⟩⟩) 4 ⟨2370⟩ 4

def event75756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10437⟩⟩) (.identity (.predecessor 0 75751 .coefficient))

def exact75757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨2377⟩⟩]⟩, (1)⟩]

theorem exact75757RawTermsValid :
    exact75757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10437⟩⟩) exact75757RawTerms (.finite 1) 75756 .exactZero (none)

def event75758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10438⟩⟩) 0 ⟨10437⟩ 75757

def event75759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10438⟩⟩) 1 ⟨6908⟩ 2

def event75760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10438⟩⟩) (.product (.predecessor 0 75758 .coefficient) (.predecessor 1 75759 .coefficient) (⟨false, false, none, none, none⟩))

def event75761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10438⟩⟩, .operator (⟨75757, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75762RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75762RawTermsValid :
    exact75762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10438⟩⟩) exact75762RawTerms .large 75760 .exactZero (none)

def event75763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10326⟩⟩) 0 ⟨10325⟩ 3083

def event75764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10326⟩⟩) 1 ⟨2370⟩ 4

def event75765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10326⟩⟩) (.sum [.predecessor 0 75763 .coefficient, .predecessor 1 75764 .coefficient])

def event75766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10326⟩⟩) (.finite 655361)

def event75767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10327⟩⟩) 0 ⟨0⟩ 20

def event75768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10327⟩⟩) 1 ⟨10325⟩ 3083

def event75769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10327⟩⟩) 2 ⟨10326⟩ 75766

def event75770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10327⟩⟩) 3 ⟨136⟩ 6

def event75771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10327⟩⟩) 4 ⟨2370⟩ 4

def event75772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10327⟩⟩) (.identity (.predecessor 0 75767 .coefficient))

def exact75773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨10694⟩⟩]⟩, (1)⟩]

theorem exact75773RawTermsValid :
    exact75773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10327⟩⟩) exact75773RawTerms (.finite 1) 75772 .exactZero (none)

def event75774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10329⟩⟩) 0 ⟨10327⟩ 75773

def event75775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10329⟩⟩) 1 ⟨7243⟩ 16137

def eventLeaf4720 : Array AnnotatedEvent := #[
  { event := event75520
    frameStart := 0 },
  { event := event75521
    frameStart := 0 },
  { event := event75522
    frameStart := 0 },
  { event := event75523
    frameStart := 0 },
  { event := event75524
    frameStart := 0 },
  { event := event75525
    frameStart := 0 },
  { event := event75526
    frameStart := 0 },
  { event := event75527
    frameStart := 0 },
  { event := event75528
    frameStart := 0 },
  { event := event75529
    frameStart := 0 },
  { event := event75530
    frameStart := 0 },
  { event := event75531
    frameStart := 0 },
  { event := event75532
    frameStart := 0 },
  { event := event75533
    frameStart := 0 },
  { event := event75534
    frameStart := 0 },
  { event := event75535
    frameStart := 0 }
]

def eventLeaf4721 : Array AnnotatedEvent := #[
  { event := event75536
    frameStart := 0 },
  { event := event75537
    frameStart := 0 },
  { event := event75538
    frameStart := 0 },
  { event := event75539
    frameStart := 0 },
  { event := event75540
    frameStart := 0 },
  { event := event75541
    frameStart := 0 },
  { event := event75542
    frameStart := 0 },
  { event := event75543
    frameStart := 0 },
  { event := event75544
    frameStart := 0 },
  { event := event75545
    frameStart := 0 },
  { event := event75546
    frameStart := 0 },
  { event := event75547
    frameStart := 0 },
  { event := event75548
    frameStart := 0 },
  { event := event75549
    frameStart := 0 },
  { event := event75550
    frameStart := 0 },
  { event := event75551
    frameStart := 0 }
]

def eventLeaf4722 : Array AnnotatedEvent := #[
  { event := event75552
    frameStart := 0 },
  { event := event75553
    frameStart := 0 },
  { event := event75554
    frameStart := 0 },
  { event := event75555
    frameStart := 0 },
  { event := event75556
    frameStart := 0 },
  { event := event75557
    frameStart := 0 },
  { event := event75558
    frameStart := 0 },
  { event := event75559
    frameStart := 0 },
  { event := event75560
    frameStart := 0 },
  { event := event75561
    frameStart := 0 },
  { event := event75562
    frameStart := 0 },
  { event := event75563
    frameStart := 0 },
  { event := event75564
    frameStart := 0 },
  { event := event75565
    frameStart := 0 },
  { event := event75566
    frameStart := 0 },
  { event := event75567
    frameStart := 0 }
]

def eventLeaf4723 : Array AnnotatedEvent := #[
  { event := event75568
    frameStart := 0 },
  { event := event75569
    frameStart := 0 },
  { event := event75570
    frameStart := 0 },
  { event := event75571
    frameStart := 0 },
  { event := event75572
    frameStart := 0 },
  { event := event75573
    frameStart := 0 },
  { event := event75574
    frameStart := 0 },
  { event := event75575
    frameStart := 0 },
  { event := event75576
    frameStart := 0 },
  { event := event75577
    frameStart := 0 },
  { event := event75578
    frameStart := 0 },
  { event := event75579
    frameStart := 0 },
  { event := event75580
    frameStart := 0 },
  { event := event75581
    frameStart := 0 },
  { event := event75582
    frameStart := 0 },
  { event := event75583
    frameStart := 0 }
]

def eventLeaf4724 : Array AnnotatedEvent := #[
  { event := event75584
    frameStart := 0 },
  { event := event75585
    frameStart := 0 },
  { event := event75586
    frameStart := 0 },
  { event := event75587
    frameStart := 0 },
  { event := event75588
    frameStart := 0 },
  { event := event75589
    frameStart := 0 },
  { event := event75590
    frameStart := 0 },
  { event := event75591
    frameStart := 0 },
  { event := event75592
    frameStart := 0 },
  { event := event75593
    frameStart := 0 },
  { event := event75594
    frameStart := 0 },
  { event := event75595
    frameStart := 0 },
  { event := event75596
    frameStart := 0 },
  { event := event75597
    frameStart := 0 },
  { event := event75598
    frameStart := 0 },
  { event := event75599
    frameStart := 0 }
]

def eventLeaf4725 : Array AnnotatedEvent := #[
  { event := event75600
    frameStart := 0 },
  { event := event75601
    frameStart := 0 },
  { event := event75602
    frameStart := 0 },
  { event := event75603
    frameStart := 0 },
  { event := event75604
    frameStart := 0 },
  { event := event75605
    frameStart := 0 },
  { event := event75606
    frameStart := 0 },
  { event := event75607
    frameStart := 0 },
  { event := event75608
    frameStart := 0 },
  { event := event75609
    frameStart := 0 },
  { event := event75610
    frameStart := 0 },
  { event := event75611
    frameStart := 0 },
  { event := event75612
    frameStart := 0 },
  { event := event75613
    frameStart := 0 },
  { event := event75614
    frameStart := 0 },
  { event := event75615
    frameStart := 0 }
]

def eventLeaf4726 : Array AnnotatedEvent := #[
  { event := event75616
    frameStart := 0 },
  { event := event75617
    frameStart := 0 },
  { event := event75618
    frameStart := 0 },
  { event := event75619
    frameStart := 0 },
  { event := event75620
    frameStart := 0 },
  { event := event75621
    frameStart := 0 },
  { event := event75622
    frameStart := 0 },
  { event := event75623
    frameStart := 0 },
  { event := event75624
    frameStart := 0 },
  { event := event75625
    frameStart := 0 },
  { event := event75626
    frameStart := 0 },
  { event := event75627
    frameStart := 0 },
  { event := event75628
    frameStart := 0 },
  { event := event75629
    frameStart := 0 },
  { event := event75630
    frameStart := 0 },
  { event := event75631
    frameStart := 0 }
]

def eventLeaf4727 : Array AnnotatedEvent := #[
  { event := event75632
    frameStart := 0 },
  { event := event75633
    frameStart := 0 },
  { event := event75634
    frameStart := 0 },
  { event := event75635
    frameStart := 0 },
  { event := event75636
    frameStart := 0 },
  { event := event75637
    frameStart := 0 },
  { event := event75638
    frameStart := 0 },
  { event := event75639
    frameStart := 0 },
  { event := event75640
    frameStart := 0 },
  { event := event75641
    frameStart := 0 },
  { event := event75642
    frameStart := 0 },
  { event := event75643
    frameStart := 0 },
  { event := event75644
    frameStart := 0 },
  { event := event75645
    frameStart := 0 },
  { event := event75646
    frameStart := 0 },
  { event := event75647
    frameStart := 0 }
]

def eventLeaf4728 : Array AnnotatedEvent := #[
  { event := event75648
    frameStart := 0 },
  { event := event75649
    frameStart := 0 },
  { event := event75650
    frameStart := 0 },
  { event := event75651
    frameStart := 0 },
  { event := event75652
    frameStart := 0 },
  { event := event75653
    frameStart := 0 },
  { event := event75654
    frameStart := 0 },
  { event := event75655
    frameStart := 0 },
  { event := event75656
    frameStart := 0 },
  { event := event75657
    frameStart := 0 },
  { event := event75658
    frameStart := 0 },
  { event := event75659
    frameStart := 0 },
  { event := event75660
    frameStart := 0 },
  { event := event75661
    frameStart := 0 },
  { event := event75662
    frameStart := 0 },
  { event := event75663
    frameStart := 0 }
]

def eventLeaf4729 : Array AnnotatedEvent := #[
  { event := event75664
    frameStart := 0 },
  { event := event75665
    frameStart := 0 },
  { event := event75666
    frameStart := 0 },
  { event := event75667
    frameStart := 0 },
  { event := event75668
    frameStart := 0 },
  { event := event75669
    frameStart := 0 },
  { event := event75670
    frameStart := 0 },
  { event := event75671
    frameStart := 0 },
  { event := event75672
    frameStart := 0 },
  { event := event75673
    frameStart := 0 },
  { event := event75674
    frameStart := 0 },
  { event := event75675
    frameStart := 0 },
  { event := event75676
    frameStart := 0 },
  { event := event75677
    frameStart := 0 },
  { event := event75678
    frameStart := 0 },
  { event := event75679
    frameStart := 0 }
]

def eventLeaf4730 : Array AnnotatedEvent := #[
  { event := event75680
    frameStart := 0 },
  { event := event75681
    frameStart := 0 },
  { event := event75682
    frameStart := 0 },
  { event := event75683
    frameStart := 0 },
  { event := event75684
    frameStart := 0 },
  { event := event75685
    frameStart := 0 },
  { event := event75686
    frameStart := 0 },
  { event := event75687
    frameStart := 0 },
  { event := event75688
    frameStart := 0 },
  { event := event75689
    frameStart := 0 },
  { event := event75690
    frameStart := 0 },
  { event := event75691
    frameStart := 0 },
  { event := event75692
    frameStart := 0 },
  { event := event75693
    frameStart := 0 },
  { event := event75694
    frameStart := 0 },
  { event := event75695
    frameStart := 0 }
]

def eventLeaf4731 : Array AnnotatedEvent := #[
  { event := event75696
    frameStart := 0 },
  { event := event75697
    frameStart := 0 },
  { event := event75698
    frameStart := 0 },
  { event := event75699
    frameStart := 0 },
  { event := event75700
    frameStart := 0 },
  { event := event75701
    frameStart := 0 },
  { event := event75702
    frameStart := 0 },
  { event := event75703
    frameStart := 0 },
  { event := event75704
    frameStart := 0 },
  { event := event75705
    frameStart := 0 },
  { event := event75706
    frameStart := 0 },
  { event := event75707
    frameStart := 0 },
  { event := event75708
    frameStart := 0 },
  { event := event75709
    frameStart := 0 },
  { event := event75710
    frameStart := 0 },
  { event := event75711
    frameStart := 0 }
]

def eventLeaf4732 : Array AnnotatedEvent := #[
  { event := event75712
    frameStart := 0 },
  { event := event75713
    frameStart := 0 },
  { event := event75714
    frameStart := 0 },
  { event := event75715
    frameStart := 0 },
  { event := event75716
    frameStart := 0 },
  { event := event75717
    frameStart := 0 },
  { event := event75718
    frameStart := 0 },
  { event := event75719
    frameStart := 0 },
  { event := event75720
    frameStart := 0 },
  { event := event75721
    frameStart := 0 },
  { event := event75722
    frameStart := 0 },
  { event := event75723
    frameStart := 0 },
  { event := event75724
    frameStart := 0 },
  { event := event75725
    frameStart := 0 },
  { event := event75726
    frameStart := 0 },
  { event := event75727
    frameStart := 0 }
]

def eventLeaf4733 : Array AnnotatedEvent := #[
  { event := event75728
    frameStart := 0 },
  { event := event75729
    frameStart := 0 },
  { event := event75730
    frameStart := 0 },
  { event := event75731
    frameStart := 0 },
  { event := event75732
    frameStart := 0 },
  { event := event75733
    frameStart := 0 },
  { event := event75734
    frameStart := 0 },
  { event := event75735
    frameStart := 0 },
  { event := event75736
    frameStart := 0 },
  { event := event75737
    frameStart := 0 },
  { event := event75738
    frameStart := 0 },
  { event := event75739
    frameStart := 0 },
  { event := event75740
    frameStart := 0 },
  { event := event75741
    frameStart := 0 },
  { event := event75742
    frameStart := 0 },
  { event := event75743
    frameStart := 0 }
]

def eventLeaf4734 : Array AnnotatedEvent := #[
  { event := event75744
    frameStart := 0 },
  { event := event75745
    frameStart := 0 },
  { event := event75746
    frameStart := 0 },
  { event := event75747
    frameStart := 0 },
  { event := event75748
    frameStart := 0 },
  { event := event75749
    frameStart := 0 },
  { event := event75750
    frameStart := 0 },
  { event := event75751
    frameStart := 0 },
  { event := event75752
    frameStart := 0 },
  { event := event75753
    frameStart := 0 },
  { event := event75754
    frameStart := 0 },
  { event := event75755
    frameStart := 0 },
  { event := event75756
    frameStart := 0 },
  { event := event75757
    frameStart := 0 },
  { event := event75758
    frameStart := 0 },
  { event := event75759
    frameStart := 0 }
]

def eventLeaf4735 : Array AnnotatedEvent := #[
  { event := event75760
    frameStart := 0 },
  { event := event75761
    frameStart := 0 },
  { event := event75762
    frameStart := 0 },
  { event := event75763
    frameStart := 0 },
  { event := event75764
    frameStart := 0 },
  { event := event75765
    frameStart := 0 },
  { event := event75766
    frameStart := 0 },
  { event := event75767
    frameStart := 0 },
  { event := event75768
    frameStart := 0 },
  { event := event75769
    frameStart := 0 },
  { event := event75770
    frameStart := 0 },
  { event := event75771
    frameStart := 0 },
  { event := event75772
    frameStart := 0 },
  { event := event75773
    frameStart := 0 },
  { event := event75774
    frameStart := 0 },
  { event := event75775
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events295
