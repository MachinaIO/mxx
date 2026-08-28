import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events190

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact48640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩]

theorem exact48640RawTermsValid :
    exact48640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48640 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6722⟩⟩) exact48640RawTerms .large 48639 .exactZero (none)

def event48641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17176⟩⟩) 0 ⟨6722⟩ 48640

def event48642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17176⟩⟩) 1 ⟨17175⟩ 48637

def event48643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17176⟩⟩) (.sum [.predecessor 0 48641 .coefficient, .predecessor 1 48642 .coefficient])

def exact48644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48644RawTermsValid :
    exact48644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48644 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17176⟩⟩) exact48644RawTerms .large 48643 .exactZero (none)

def event48645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27891⟩⟩) 0 ⟨17176⟩ 48644

def event48646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27891⟩⟩) 1 ⟨27886⟩ 48629

def event48647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27891⟩⟩) (.sum [.predecessor 0 48645 .coefficient, .predecessor 1 48646 .coefficient])

def exact48648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48648RawTermsValid :
    exact48648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48648 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27891⟩⟩) exact48648RawTerms .large 48647 .exactZero (none)

def event48649 : Event := .preFoldPolynomial 48648 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event48650 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27891⟩⟩) 48649 exact48650RawTerms .large 48647 .exactZero (none)

def event48651 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15949⟩⟩) ⟨⟨135⟩, ⟨42⟩, ⟨109⟩⟩ ⟨48493, 48651⟩

def event48652 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21339⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩) (1) 0 2 (.universal 48651 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩) (none) 48650)

def event48653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21339⟩⟩, .relation 48652 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩)

def event48654 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21339⟩⟩, .relation 48652 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩)

def event48655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21339⟩⟩, .relation 48652 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩)

def event48656 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21339⟩⟩, .relation 48652 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48657RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48657RawTermsValid :
    exact48657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48657 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21339⟩⟩) exact48657RawTerms .large 48489 (.finite 1811303510016) (some (48491))

def event48658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27888⟩⟩) 0 ⟨21339⟩ 48657

def event48659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27888⟩⟩) 1 ⟨27887⟩ 48479

def event48660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27888⟩⟩) (.sum [.predecessor 0 48658 .coefficient, .predecessor 1 48659 .coefficient])

def event48661 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27888⟩⟩, .operator (⟨48657, 0⟩, ⟨48479, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩, (1)⟩)

def event48662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27888⟩⟩, .operator (⟨48657, 2⟩, ⟨48479, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], [⟨.program ⟨214⟩, ⟨24167⟩⟩]⟩, (-1)⟩)

def event48663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27888⟩⟩) (.sum [.result 48657 .summary, .result 48479 .summary])

def exact48664RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48664RawTermsValid :
    exact48664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48664 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27888⟩⟩) exact48664RawTerms .large 48660 (.finite 1292068473939586330624) (some (48663))

def event48665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27889⟩⟩) 0 ⟨27888⟩ 48664

def event48666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27889⟩⟩) 1 ⟨6642⟩ 5719

def event48667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27889⟩⟩) (.product (.predecessor 0 48665 .coefficient) (.predecessor 1 48666 .coefficient) (⟨false, false, none, none, none⟩))

def event48668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27889⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) [⟨.result 5715 .coefficient, false, none⟩])

def event48669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27889⟩⟩) (.product (.result 48664 .summary) (.transfer 48668) (⟨false, false, none, none, none⟩))

def event48670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27889⟩⟩, .operator (⟨48664, 0⟩, ⟨5719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩)

def event48671 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27889⟩⟩, .operator (⟨48664, 1⟩, ⟨5719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (-1)⟩)

def event48672 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27889⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6641⟩⟩) ⟨6592⟩ 5712)

def event48673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27889⟩⟩, .relation 48672 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48674RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48674RawTermsValid :
    exact48674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48674 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27889⟩⟩) exact48674RawTerms .large 48667 (.finite 4741911972453864866771369984) (some (48669))

def event48675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24104⟩⟩) 0 ⟨6689⟩ 5477

def event48676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24104⟩⟩) 1 ⟨24103⟩ 41341

def event48677 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24104⟩⟩) (.authority (.operator))

def exact48678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩]

theorem exact48678RawTermsValid :
    exact48678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48678 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24104⟩⟩) exact48678RawTerms .large 48677 .exactZero (none)

def event48679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27668⟩⟩) 0 ⟨24104⟩ 48678

def event48680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27668⟩⟩) (.authority (.operator))

def exact48681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩]

theorem exact48681RawTermsValid :
    exact48681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48681 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27668⟩⟩) exact48681RawTerms (.finite 8192) 48680 .exactZero (none)

def event48682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27670⟩⟩) 0 ⟨26001⟩ 41625

def event48683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27670⟩⟩) 1 ⟨27668⟩ 48681

def event48684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27670⟩⟩) (.product (.predecessor 0 48682 .coefficient) (.predecessor 1 48683 .coefficient) (⟨false, false, none, none, none⟩))

def event48685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27670⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩) [⟨.result 48681 .coefficient, false, none⟩])

def event48686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27670⟩⟩) (.product (.result 41625 .summary) (.transfer 48685) (⟨false, false, none, none, none⟩))

def event48687 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27670⟩⟩, .operator (⟨41625, 0⟩, ⟨48681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩)

def event48688 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27670⟩⟩, .operator (⟨41625, 1⟩, ⟨48681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩)

def event48689 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27670⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27668⟩⟩) ⟨24104⟩ 48678)

def event48690 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27670⟩⟩, .relation 48689 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (-1)⟩)

def exact48691RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (-1)⟩]

theorem exact48691RawTermsValid :
    exact48691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48691 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27670⟩⟩) exact48691RawTerms .large 48684 (.finite 1292046059683262234624) (some (48686))

def event48692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21192⟩⟩) 0 ⟨15830⟩ 1860

def event48693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21192⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact48694RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩]

theorem exact48694RawTermsValid :
    exact48694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48694 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21192⟩⟩) exact48694RawTerms (.finite 136065468) 48693 .exactZero (none)

def event48695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21194⟩⟩) 0 ⟨21192⟩ 48694

def event48696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21194⟩⟩) 1 ⟨2348⟩ 4

def event48697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21194⟩⟩) (.scale (.predecessor 0 48695 .coefficient) (.value (.predecessor 1 48696 .coefficient)))

def exact48698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩]

theorem exact48698RawTermsValid :
    exact48698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48698 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21194⟩⟩) exact48698RawTerms (.finite 136065468) 48697 .exactZero (none)

def event48699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21195⟩⟩) 0 ⟨5553⟩ 36137

def event48700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21195⟩⟩) 1 ⟨21194⟩ 48698

def event48701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21195⟩⟩) (.product (.predecessor 0 48699 .coefficient) (.predecessor 1 48700 .coefficient) (⟨false, false, none, none, none⟩))

def event48702 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21195⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩) [⟨.result 48694 .coefficient, false, none⟩])

def event48703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21195⟩⟩) (.product (.result 36137 .summary) (.transfer 48702) (⟨false, false, none, none, none⟩))

def event48704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21195⟩⟩, .operator (⟨36137, 0⟩, ⟨48698, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩)

def event48705 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21193⟩⟩)

def event48706 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48707 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48709 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48711 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48713 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48713

def event48715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48711

def event48716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48714 .coefficient) (.value (.predecessor 1 48715 .coefficient)))

def event48717 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48717

def event48719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48709

def event48720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48718 .coefficient, .predecessor 1 48719 .coefficient])

def event48721 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48721

def event48723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48707

def event48724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48723 .coefficient))

def event48725 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11393⟩⟩) 0 ⟨5548⟩ 48725

def event48727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11393⟩⟩) (.authority (.programFamilyFact))

def exact48728RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩], []⟩, (1)⟩]

theorem exact48728RawTermsValid :
    exact48728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11393⟩⟩) exact48728RawTerms (.finite 16) 48727 .exactZero (none)

def event48729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14008⟩⟩) 0 ⟨5548⟩ 48725

def event48730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14008⟩⟩) (.authority (.programFamilyFact))

def exact48731RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14008⟩⟩], []⟩, (1)⟩]

theorem exact48731RawTermsValid :
    exact48731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48731 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14008⟩⟩) exact48731RawTerms (.finite 16) 48730 .exactZero (none)

def event48732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14009⟩⟩) 0 ⟨14008⟩ 48731

def event48733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14009⟩⟩) 1 ⟨11393⟩ 48728

def event48734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14009⟩⟩) (.product (.predecessor 0 48732 .coefficient) (.predecessor 1 48733 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14009⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], []⟩) [⟨.result 48731 .coefficient, true, some 1⟩, ⟨.result 48728 .coefficient, true, some 1⟩])

def event48736 : Event := .survivorFold (1) 48735

def exact48737RawTerms : List Term := []

theorem exact48737RawTermsValid :
    exact48737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48737 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14009⟩⟩) exact48737RawTerms (.finite 256) 48734 (.finite 256) (some (48735))

def event48738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14010⟩⟩) 0 ⟨14009⟩ 48737

def event48739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14010⟩⟩) (.identity (.predecessor 0 48738 .coefficient))

def event48740 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14010⟩⟩) (.finite 256)

def event48741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15829⟩⟩) 0 ⟨14010⟩ 48740

def event48742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15829⟩⟩) (.authority (.programFamilyFact))

def exact48743RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], []⟩, (1)⟩]

theorem exact48743RawTermsValid :
    exact48743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48743 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15829⟩⟩) exact48743RawTerms (.finite 16) 48742 .exactZero (none)

def event48744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15830⟩⟩) 0 ⟨15829⟩ 48743

def event48745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15830⟩⟩) (.identity (.predecessor 0 48744 .coefficient))

def event48746 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15830⟩⟩) (.finite 16)

def event48747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21192⟩⟩) 0 ⟨15830⟩ 48746

def event48748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21192⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact48749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩]

theorem exact48749RawTermsValid :
    exact48749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48749 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21192⟩⟩) exact48749RawTerms (.finite 136065468) 48748 .exactZero (none)

def event48750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact48751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact48751RawTermsValid :
    exact48751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact48751RawTerms .large 48750 .exactZero (none)

def event48752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21193⟩⟩) 0 ⟨6⟩ 48751

def event48753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21193⟩⟩) 1 ⟨21192⟩ 48749

def event48754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21193⟩⟩) (.product (.predecessor 0 48752 .coefficient) (.predecessor 1 48753 .coefficient) (⟨false, false, none, none, none⟩))

def event48755 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21193⟩⟩, .operator (⟨48751, 0⟩, ⟨48749, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩)

def exact48756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩]

theorem exact48756RawTermsValid :
    exact48756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48756 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21193⟩⟩) exact48756RawTerms .large 48754 .exactZero (none)

def event48757 : Event := .preFoldPolynomial 48756 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩] .exactZero none

def exact48758RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩, (1)⟩]

def event48758 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21193⟩⟩) 48757 exact48758RawTerms .large 48754 .exactZero (none)

def event48759 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27674⟩⟩)

def event48760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48761 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48763 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48765 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48767 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48767

def event48769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48765

def event48770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48768 .coefficient) (.value (.predecessor 1 48769 .coefficient)))

def event48771 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48771

def event48773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48763

def event48774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48772 .coefficient, .predecessor 1 48773 .coefficient])

def event48775 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48775

def event48777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48761

def event48778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48777 .coefficient))

def event48779 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11393⟩⟩) 0 ⟨5548⟩ 48779

def event48781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11393⟩⟩) (.authority (.programFamilyFact))

def exact48782RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩], []⟩, (1)⟩]

theorem exact48782RawTermsValid :
    exact48782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48782 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11393⟩⟩) exact48782RawTerms (.finite 16) 48781 .exactZero (none)

def event48783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14008⟩⟩) 0 ⟨5548⟩ 48779

def event48784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14008⟩⟩) (.authority (.programFamilyFact))

def exact48785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14008⟩⟩], []⟩, (1)⟩]

theorem exact48785RawTermsValid :
    exact48785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14008⟩⟩) exact48785RawTerms (.finite 16) 48784 .exactZero (none)

def event48786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14009⟩⟩) 0 ⟨14008⟩ 48785

def event48787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14009⟩⟩) 1 ⟨11393⟩ 48782

def event48788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14009⟩⟩) (.product (.predecessor 0 48786 .coefficient) (.predecessor 1 48787 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48789 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14009⟩⟩, .operator (⟨48785, 0⟩, ⟨48782, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], []⟩, (1)⟩)

def exact48790RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11393⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], []⟩, (1)⟩]

theorem exact48790RawTermsValid :
    exact48790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48790 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14009⟩⟩) exact48790RawTerms (.finite 256) 48788 .exactZero (none)

def event48791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14010⟩⟩) 0 ⟨14009⟩ 48790

def event48792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14010⟩⟩) (.identity (.predecessor 0 48791 .coefficient))

def event48793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14010⟩⟩) (.finite 256)

def event48794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15829⟩⟩) 0 ⟨14010⟩ 48793

def event48795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15829⟩⟩) (.authority (.programFamilyFact))

def exact48796RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], []⟩, (1)⟩]

theorem exact48796RawTermsValid :
    exact48796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15829⟩⟩) exact48796RawTerms (.finite 16) 48795 .exactZero (none)

def event48797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15830⟩⟩) 0 ⟨15829⟩ 48796

def event48798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15830⟩⟩) (.identity (.predecessor 0 48797 .coefficient))

def event48799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15830⟩⟩) (.finite 16)

def event48800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24103⟩⟩) 0 ⟨15830⟩ 48799

def event48801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24103⟩⟩) (.authority (.programFamilyFact))

def event48802 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24103⟩⟩) (.finite 3720)

def event48803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event48804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24104⟩⟩) 0 ⟨6689⟩ 48803

def event48805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24104⟩⟩) 1 ⟨24103⟩ 48802

def event48806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24104⟩⟩) (.authority (.operator))

def exact48807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩]

theorem exact48807RawTermsValid :
    exact48807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24104⟩⟩) exact48807RawTerms .large 48806 .exactZero (none)

def event48808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27668⟩⟩) 0 ⟨24104⟩ 48807

def event48809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27668⟩⟩) (.authority (.operator))

def exact48810RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩]

theorem exact48810RawTermsValid :
    exact48810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27668⟩⟩) exact48810RawTerms (.finite 8192) 48809 .exactZero (none)

def event48811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event48812 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event48813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15904⟩⟩) 0 ⟨15830⟩ 48799

def event48814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15904⟩⟩) 1 ⟨110⟩ 48812

def event48815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15904⟩⟩) (.sum [.predecessor 0 48813 .coefficient, .predecessor 1 48814 .coefficient])

def event48816 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15904⟩⟩) (.finite 16)

def event48817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15905⟩⟩) 0 ⟨15904⟩ 48816

def event48818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15905⟩⟩) (.identity (.predecessor 0 48817 .coefficient))

def exact48819RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], []⟩, (1)⟩]

theorem exact48819RawTermsValid :
    exact48819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15905⟩⟩) exact48819RawTerms (.finite 16) 48818 .exactZero (none)

def event48820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact48821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48821RawTermsValid :
    exact48821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact48821RawTerms .large 48820 .exactZero (none)

def event48822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15906⟩⟩) 0 ⟨6544⟩ 48821

def event48823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15906⟩⟩) 1 ⟨15905⟩ 48819

def event48824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15906⟩⟩) (.product (.predecessor 0 48822 .coefficient) (.predecessor 1 48823 .coefficient) (⟨false, false, none, none, none⟩))

def event48825 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15906⟩⟩, .operator (⟨48821, 0⟩, ⟨48819, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48826RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48826RawTermsValid :
    exact48826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15906⟩⟩) exact48826RawTerms .large 48824 .exactZero (none)

def event48827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 48803

def event48828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact48829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact48829RawTermsValid :
    exact48829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact48829RawTerms .large 48828 .exactZero (none)

def event48830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15907⟩⟩) 0 ⟨6696⟩ 48829

def event48831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15907⟩⟩) 1 ⟨15906⟩ 48826

def event48832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15907⟩⟩) (.sum [.predecessor 0 48830 .coefficient, .predecessor 1 48831 .coefficient])

def exact48833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48833RawTermsValid :
    exact48833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15907⟩⟩) exact48833RawTerms .large 48832 .exactZero (none)

def event48834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27669⟩⟩) 0 ⟨15907⟩ 48833

def event48835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27669⟩⟩) 1 ⟨27668⟩ 48810

def event48836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27669⟩⟩) (.product (.predecessor 0 48834 .coefficient) (.predecessor 1 48835 .coefficient) (⟨false, false, none, none, none⟩))

def event48837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27669⟩⟩, .operator (⟨48833, 0⟩, ⟨48810, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩)

def event48838 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27669⟩⟩, .operator (⟨48833, 1⟩, ⟨48810, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩)

def event48839 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27669⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27668⟩⟩) ⟨24104⟩ 48807)

def event48840 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27669⟩⟩, .relation 48839 0, ⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (-1)⟩)

def exact48841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (-1)⟩]

theorem exact48841RawTermsValid :
    exact48841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27669⟩⟩) exact48841RawTerms .large 48836 .exactZero (none)

def event48842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17229⟩⟩) 0 ⟨15830⟩ 48799

def event48843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17229⟩⟩) (.authority (.programFamilyFact))

def exact48844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩, (1)⟩]

theorem exact48844RawTermsValid :
    exact48844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17229⟩⟩) exact48844RawTerms (.finite 16) 48843 .exactZero (none)

def event48845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17231⟩⟩) 0 ⟨6544⟩ 48821

def event48846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17231⟩⟩) 1 ⟨17229⟩ 48844

def event48847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17231⟩⟩) (.product (.predecessor 0 48845 .coefficient) (.predecessor 1 48846 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17231⟩⟩, .operator (⟨48821, 0⟩, ⟨48844, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48849RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48849RawTermsValid :
    exact48849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17231⟩⟩) exact48849RawTerms .large 48847 .exactZero (none)

def event48850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6720⟩⟩) 0 ⟨6689⟩ 48803

def event48851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6720⟩⟩) (.authority (.operator))

def exact48852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩]

theorem exact48852RawTermsValid :
    exact48852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6720⟩⟩) exact48852RawTerms .large 48851 .exactZero (none)

def event48853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17232⟩⟩) 0 ⟨6720⟩ 48852

def event48854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17232⟩⟩) 1 ⟨17231⟩ 48849

def event48855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17232⟩⟩) (.sum [.predecessor 0 48853 .coefficient, .predecessor 1 48854 .coefficient])

def exact48856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48856RawTermsValid :
    exact48856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17232⟩⟩) exact48856RawTerms .large 48855 .exactZero (none)

def event48857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27674⟩⟩) 0 ⟨17232⟩ 48856

def event48858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27674⟩⟩) 1 ⟨27669⟩ 48841

def event48859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27674⟩⟩) (.sum [.predecessor 0 48857 .coefficient, .predecessor 1 48858 .coefficient])

def exact48860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48860RawTermsValid :
    exact48860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27674⟩⟩) exact48860RawTerms .large 48859 .exactZero (none)

def event48861 : Event := .preFoldPolynomial 48860 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event48862 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27674⟩⟩) 48861 exact48862RawTerms .large 48859 .exactZero (none)

def event48863 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15830⟩⟩) ⟨⟨133⟩, ⟨40⟩, ⟨109⟩⟩ ⟨48705, 48863⟩

def event48864 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21195⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩) (1) 0 2 (.universal 48863 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21192⟩⟩]⟩) (none) 48862)

def event48865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21195⟩⟩, .relation 48864 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩)

def event48866 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21195⟩⟩, .relation 48864 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩)

def event48867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21195⟩⟩, .relation 48864 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩)

def event48868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21195⟩⟩, .relation 48864 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48869RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48869RawTermsValid :
    exact48869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21195⟩⟩) exact48869RawTerms .large 48701 (.finite 1811303510016) (some (48703))

def event48870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27671⟩⟩) 0 ⟨21195⟩ 48869

def event48871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27671⟩⟩) 1 ⟨27670⟩ 48691

def event48872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27671⟩⟩) (.sum [.predecessor 0 48870 .coefficient, .predecessor 1 48871 .coefficient])

def event48873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27671⟩⟩, .operator (⟨48869, 0⟩, ⟨48691, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27668⟩⟩]⟩, (1)⟩)

def event48874 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27671⟩⟩, .operator (⟨48869, 2⟩, ⟨48691, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15829⟩⟩], [⟨.program ⟨214⟩, ⟨24104⟩⟩]⟩, (-1)⟩)

def event48875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27671⟩⟩) (.sum [.result 48869 .summary, .result 48691 .summary])

def exact48876RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48876RawTermsValid :
    exact48876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48876 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27671⟩⟩) exact48876RawTerms .large 48872 (.finite 1292046061494565744640) (some (48875))

def event48877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27672⟩⟩) 0 ⟨27671⟩ 48876

def event48878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27672⟩⟩) 1 ⟨6644⟩ 5739

def event48879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27672⟩⟩) (.product (.predecessor 0 48877 .coefficient) (.predecessor 1 48878 .coefficient) (⟨false, false, none, none, none⟩))

def event48880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27672⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) [⟨.result 5735 .coefficient, false, none⟩])

def event48881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27672⟩⟩) (.product (.result 48876 .summary) (.transfer 48880) (⟨false, false, none, none, none⟩))

def event48882 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27672⟩⟩, .operator (⟨48876, 0⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩)

def event48883 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27672⟩⟩, .operator (⟨48876, 1⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (-1)⟩)

def event48884 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27672⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6643⟩⟩) ⟨6593⟩ 5732)

def event48885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27672⟩⟩, .relation 48884 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48886RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48886RawTermsValid :
    exact48886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48886 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27672⟩⟩) exact48886RawTerms .large 48879 (.finite 4741829718422040195880714240) (some (48881))

def event48887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24041⟩⟩) 0 ⟨6689⟩ 5477

def event48888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24041⟩⟩) 1 ⟨24040⟩ 41823

def event48889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24041⟩⟩) (.authority (.operator))

def exact48890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24041⟩⟩]⟩, (1)⟩]

theorem exact48890RawTermsValid :
    exact48890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48890 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24041⟩⟩) exact48890RawTerms .large 48889 .exactZero (none)

def event48891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27451⟩⟩) 0 ⟨24041⟩ 48890

def event48892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27451⟩⟩) (.authority (.operator))

def exact48893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27451⟩⟩]⟩, (1)⟩]

theorem exact48893RawTermsValid :
    exact48893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27451⟩⟩) exact48893RawTerms (.finite 8192) 48892 .exactZero (none)

def event48894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27453⟩⟩) 0 ⟨25924⟩ 42107

def event48895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27453⟩⟩) 1 ⟨27451⟩ 48893

def eventLeaf3040 : Array AnnotatedEvent := #[
  { event := event48640
    frameStart := 48547 },
  { event := event48641
    frameStart := 48547 },
  { event := event48642
    frameStart := 48547 },
  { event := event48643
    frameStart := 48547 },
  { event := event48644
    frameStart := 48547 },
  { event := event48645
    frameStart := 48547 },
  { event := event48646
    frameStart := 48547 },
  { event := event48647
    frameStart := 48547 },
  { event := event48648
    frameStart := 48547 },
  { event := event48649
    frameStart := 48547 },
  { event := event48650
    frameStart := 48547 },
  { event := event48651
    frameStart := 0 },
  { event := event48652
    frameStart := 0 },
  { event := event48653
    frameStart := 0 },
  { event := event48654
    frameStart := 0 },
  { event := event48655
    frameStart := 0 }
]

def eventLeaf3041 : Array AnnotatedEvent := #[
  { event := event48656
    frameStart := 0 },
  { event := event48657
    frameStart := 0 },
  { event := event48658
    frameStart := 0 },
  { event := event48659
    frameStart := 0 },
  { event := event48660
    frameStart := 0 },
  { event := event48661
    frameStart := 0 },
  { event := event48662
    frameStart := 0 },
  { event := event48663
    frameStart := 0 },
  { event := event48664
    frameStart := 0 },
  { event := event48665
    frameStart := 0 },
  { event := event48666
    frameStart := 0 },
  { event := event48667
    frameStart := 0 },
  { event := event48668
    frameStart := 0 },
  { event := event48669
    frameStart := 0 },
  { event := event48670
    frameStart := 0 },
  { event := event48671
    frameStart := 0 }
]

def eventLeaf3042 : Array AnnotatedEvent := #[
  { event := event48672
    frameStart := 0 },
  { event := event48673
    frameStart := 0 },
  { event := event48674
    frameStart := 0 },
  { event := event48675
    frameStart := 0 },
  { event := event48676
    frameStart := 0 },
  { event := event48677
    frameStart := 0 },
  { event := event48678
    frameStart := 0 },
  { event := event48679
    frameStart := 0 },
  { event := event48680
    frameStart := 0 },
  { event := event48681
    frameStart := 0 },
  { event := event48682
    frameStart := 0 },
  { event := event48683
    frameStart := 0 },
  { event := event48684
    frameStart := 0 },
  { event := event48685
    frameStart := 0 },
  { event := event48686
    frameStart := 0 },
  { event := event48687
    frameStart := 0 }
]

def eventLeaf3043 : Array AnnotatedEvent := #[
  { event := event48688
    frameStart := 0 },
  { event := event48689
    frameStart := 0 },
  { event := event48690
    frameStart := 0 },
  { event := event48691
    frameStart := 0 },
  { event := event48692
    frameStart := 0 },
  { event := event48693
    frameStart := 0 },
  { event := event48694
    frameStart := 0 },
  { event := event48695
    frameStart := 0 },
  { event := event48696
    frameStart := 0 },
  { event := event48697
    frameStart := 0 },
  { event := event48698
    frameStart := 0 },
  { event := event48699
    frameStart := 0 },
  { event := event48700
    frameStart := 0 },
  { event := event48701
    frameStart := 0 },
  { event := event48702
    frameStart := 0 },
  { event := event48703
    frameStart := 0 }
]

def eventLeaf3044 : Array AnnotatedEvent := #[
  { event := event48704
    frameStart := 0 },
  { event := event48705
    frameStart := 48705 },
  { event := event48706
    frameStart := 48705 },
  { event := event48707
    frameStart := 48705 },
  { event := event48708
    frameStart := 48705 },
  { event := event48709
    frameStart := 48705 },
  { event := event48710
    frameStart := 48705 },
  { event := event48711
    frameStart := 48705 },
  { event := event48712
    frameStart := 48705 },
  { event := event48713
    frameStart := 48705 },
  { event := event48714
    frameStart := 48705 },
  { event := event48715
    frameStart := 48705 },
  { event := event48716
    frameStart := 48705 },
  { event := event48717
    frameStart := 48705 },
  { event := event48718
    frameStart := 48705 },
  { event := event48719
    frameStart := 48705 }
]

def eventLeaf3045 : Array AnnotatedEvent := #[
  { event := event48720
    frameStart := 48705 },
  { event := event48721
    frameStart := 48705 },
  { event := event48722
    frameStart := 48705 },
  { event := event48723
    frameStart := 48705 },
  { event := event48724
    frameStart := 48705 },
  { event := event48725
    frameStart := 48705 },
  { event := event48726
    frameStart := 48705 },
  { event := event48727
    frameStart := 48705 },
  { event := event48728
    frameStart := 48705 },
  { event := event48729
    frameStart := 48705 },
  { event := event48730
    frameStart := 48705 },
  { event := event48731
    frameStart := 48705 },
  { event := event48732
    frameStart := 48705 },
  { event := event48733
    frameStart := 48705 },
  { event := event48734
    frameStart := 48705 },
  { event := event48735
    frameStart := 48705 }
]

def eventLeaf3046 : Array AnnotatedEvent := #[
  { event := event48736
    frameStart := 48705 },
  { event := event48737
    frameStart := 48705 },
  { event := event48738
    frameStart := 48705 },
  { event := event48739
    frameStart := 48705 },
  { event := event48740
    frameStart := 48705 },
  { event := event48741
    frameStart := 48705 },
  { event := event48742
    frameStart := 48705 },
  { event := event48743
    frameStart := 48705 },
  { event := event48744
    frameStart := 48705 },
  { event := event48745
    frameStart := 48705 },
  { event := event48746
    frameStart := 48705 },
  { event := event48747
    frameStart := 48705 },
  { event := event48748
    frameStart := 48705 },
  { event := event48749
    frameStart := 48705 },
  { event := event48750
    frameStart := 48705 },
  { event := event48751
    frameStart := 48705 }
]

def eventLeaf3047 : Array AnnotatedEvent := #[
  { event := event48752
    frameStart := 48705 },
  { event := event48753
    frameStart := 48705 },
  { event := event48754
    frameStart := 48705 },
  { event := event48755
    frameStart := 48705 },
  { event := event48756
    frameStart := 48705 },
  { event := event48757
    frameStart := 48705 },
  { event := event48758
    frameStart := 48705 },
  { event := event48759
    frameStart := 48759 },
  { event := event48760
    frameStart := 48759 },
  { event := event48761
    frameStart := 48759 },
  { event := event48762
    frameStart := 48759 },
  { event := event48763
    frameStart := 48759 },
  { event := event48764
    frameStart := 48759 },
  { event := event48765
    frameStart := 48759 },
  { event := event48766
    frameStart := 48759 },
  { event := event48767
    frameStart := 48759 }
]

def eventLeaf3048 : Array AnnotatedEvent := #[
  { event := event48768
    frameStart := 48759 },
  { event := event48769
    frameStart := 48759 },
  { event := event48770
    frameStart := 48759 },
  { event := event48771
    frameStart := 48759 },
  { event := event48772
    frameStart := 48759 },
  { event := event48773
    frameStart := 48759 },
  { event := event48774
    frameStart := 48759 },
  { event := event48775
    frameStart := 48759 },
  { event := event48776
    frameStart := 48759 },
  { event := event48777
    frameStart := 48759 },
  { event := event48778
    frameStart := 48759 },
  { event := event48779
    frameStart := 48759 },
  { event := event48780
    frameStart := 48759 },
  { event := event48781
    frameStart := 48759 },
  { event := event48782
    frameStart := 48759 },
  { event := event48783
    frameStart := 48759 }
]

def eventLeaf3049 : Array AnnotatedEvent := #[
  { event := event48784
    frameStart := 48759 },
  { event := event48785
    frameStart := 48759 },
  { event := event48786
    frameStart := 48759 },
  { event := event48787
    frameStart := 48759 },
  { event := event48788
    frameStart := 48759 },
  { event := event48789
    frameStart := 48759 },
  { event := event48790
    frameStart := 48759 },
  { event := event48791
    frameStart := 48759 },
  { event := event48792
    frameStart := 48759 },
  { event := event48793
    frameStart := 48759 },
  { event := event48794
    frameStart := 48759 },
  { event := event48795
    frameStart := 48759 },
  { event := event48796
    frameStart := 48759 },
  { event := event48797
    frameStart := 48759 },
  { event := event48798
    frameStart := 48759 },
  { event := event48799
    frameStart := 48759 }
]

def eventLeaf3050 : Array AnnotatedEvent := #[
  { event := event48800
    frameStart := 48759 },
  { event := event48801
    frameStart := 48759 },
  { event := event48802
    frameStart := 48759 },
  { event := event48803
    frameStart := 48759 },
  { event := event48804
    frameStart := 48759 },
  { event := event48805
    frameStart := 48759 },
  { event := event48806
    frameStart := 48759 },
  { event := event48807
    frameStart := 48759 },
  { event := event48808
    frameStart := 48759 },
  { event := event48809
    frameStart := 48759 },
  { event := event48810
    frameStart := 48759 },
  { event := event48811
    frameStart := 48759 },
  { event := event48812
    frameStart := 48759 },
  { event := event48813
    frameStart := 48759 },
  { event := event48814
    frameStart := 48759 },
  { event := event48815
    frameStart := 48759 }
]

def eventLeaf3051 : Array AnnotatedEvent := #[
  { event := event48816
    frameStart := 48759 },
  { event := event48817
    frameStart := 48759 },
  { event := event48818
    frameStart := 48759 },
  { event := event48819
    frameStart := 48759 },
  { event := event48820
    frameStart := 48759 },
  { event := event48821
    frameStart := 48759 },
  { event := event48822
    frameStart := 48759 },
  { event := event48823
    frameStart := 48759 },
  { event := event48824
    frameStart := 48759 },
  { event := event48825
    frameStart := 48759 },
  { event := event48826
    frameStart := 48759 },
  { event := event48827
    frameStart := 48759 },
  { event := event48828
    frameStart := 48759 },
  { event := event48829
    frameStart := 48759 },
  { event := event48830
    frameStart := 48759 },
  { event := event48831
    frameStart := 48759 }
]

def eventLeaf3052 : Array AnnotatedEvent := #[
  { event := event48832
    frameStart := 48759 },
  { event := event48833
    frameStart := 48759 },
  { event := event48834
    frameStart := 48759 },
  { event := event48835
    frameStart := 48759 },
  { event := event48836
    frameStart := 48759 },
  { event := event48837
    frameStart := 48759 },
  { event := event48838
    frameStart := 48759 },
  { event := event48839
    frameStart := 48759 },
  { event := event48840
    frameStart := 48759 },
  { event := event48841
    frameStart := 48759 },
  { event := event48842
    frameStart := 48759 },
  { event := event48843
    frameStart := 48759 },
  { event := event48844
    frameStart := 48759 },
  { event := event48845
    frameStart := 48759 },
  { event := event48846
    frameStart := 48759 },
  { event := event48847
    frameStart := 48759 }
]

def eventLeaf3053 : Array AnnotatedEvent := #[
  { event := event48848
    frameStart := 48759 },
  { event := event48849
    frameStart := 48759 },
  { event := event48850
    frameStart := 48759 },
  { event := event48851
    frameStart := 48759 },
  { event := event48852
    frameStart := 48759 },
  { event := event48853
    frameStart := 48759 },
  { event := event48854
    frameStart := 48759 },
  { event := event48855
    frameStart := 48759 },
  { event := event48856
    frameStart := 48759 },
  { event := event48857
    frameStart := 48759 },
  { event := event48858
    frameStart := 48759 },
  { event := event48859
    frameStart := 48759 },
  { event := event48860
    frameStart := 48759 },
  { event := event48861
    frameStart := 48759 },
  { event := event48862
    frameStart := 48759 },
  { event := event48863
    frameStart := 0 }
]

def eventLeaf3054 : Array AnnotatedEvent := #[
  { event := event48864
    frameStart := 0 },
  { event := event48865
    frameStart := 0 },
  { event := event48866
    frameStart := 0 },
  { event := event48867
    frameStart := 0 },
  { event := event48868
    frameStart := 0 },
  { event := event48869
    frameStart := 0 },
  { event := event48870
    frameStart := 0 },
  { event := event48871
    frameStart := 0 },
  { event := event48872
    frameStart := 0 },
  { event := event48873
    frameStart := 0 },
  { event := event48874
    frameStart := 0 },
  { event := event48875
    frameStart := 0 },
  { event := event48876
    frameStart := 0 },
  { event := event48877
    frameStart := 0 },
  { event := event48878
    frameStart := 0 },
  { event := event48879
    frameStart := 0 }
]

def eventLeaf3055 : Array AnnotatedEvent := #[
  { event := event48880
    frameStart := 0 },
  { event := event48881
    frameStart := 0 },
  { event := event48882
    frameStart := 0 },
  { event := event48883
    frameStart := 0 },
  { event := event48884
    frameStart := 0 },
  { event := event48885
    frameStart := 0 },
  { event := event48886
    frameStart := 0 },
  { event := event48887
    frameStart := 0 },
  { event := event48888
    frameStart := 0 },
  { event := event48889
    frameStart := 0 },
  { event := event48890
    frameStart := 0 },
  { event := event48891
    frameStart := 0 },
  { event := event48892
    frameStart := 0 },
  { event := event48893
    frameStart := 0 },
  { event := event48894
    frameStart := 0 },
  { event := event48895
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events190
