import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events066

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact16896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact16896RawTermsValid :
    exact16896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6811⟩⟩) exact16896RawTerms .large 16895 .exactZero (none)

def event16897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18666⟩⟩) 0 ⟨6811⟩ 16896

def event16898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18666⟩⟩) 1 ⟨18665⟩ 16774

def event16899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18666⟩⟩) (.sum [.predecessor 0 16897 .coefficient, .predecessor 1 16898 .coefficient])

def exact16900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact16900RawTermsValid :
    exact16900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18666⟩⟩) exact16900RawTerms .large 16899 .exactZero (none)

def event16901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18694⟩⟩) 0 ⟨18666⟩ 16900

def event16902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18694⟩⟩) 1 ⟨18693⟩ 16741

def event16903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18694⟩⟩) (.product (.predecessor 0 16901 .coefficient) (.predecessor 1 16902 .coefficient) (⟨false, false, none, none, none⟩))

def event16904 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 33⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16905 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16906 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16905 0, ⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16907 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 17⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16908 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 29⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16909 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16910 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16909 0, ⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 16⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16912 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 28⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16913 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16914 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16913 0, ⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16915 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 15⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16916 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 27⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16917 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16917 0, ⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16919 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 14⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16920 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 34⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16921 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16922 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16921 0, ⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 13⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16924 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 32⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16925 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16925 0, ⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16927 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 12⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16928 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 30⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16929 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16930 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16929 0, ⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16931 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 11⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16932 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 26⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16933 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16934 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16933 0, ⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16935 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 10⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16936 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 35⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16937 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16938 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16937 0, ⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16939 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 9⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16940 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 25⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16941 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16942 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16941 0, ⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16943 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 8⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16944 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 24⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16945 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16946 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16945 0, ⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 7⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 23⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16949 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16950 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16949 0, ⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16951 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 6⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16952 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 22⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16953 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16953 0, ⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 5⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16956 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 21⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16957 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16958 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16957 0, ⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16959 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 4⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16960 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 31⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16961 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16961 0, ⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 3⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16964 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 20⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16965 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16966 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16965 0, ⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16967 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 2⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16968 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 19⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16969 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16970 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16969 0, ⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 1⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event16972 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 18⟩, ⟨16741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event16973 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18694⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 16738)

def event16974 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .relation 16973 0, ⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event16975 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18694⟩⟩, .operator (⟨16900, 0⟩, ⟨16741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def exact16976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩]

theorem exact16976RawTermsValid :
    exact16976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18694⟩⟩) exact16976RawTerms .large 16903 .exactZero (none)

def event16977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18511⟩⟩) 0 ⟨18402⟩ 16730

def event16978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18511⟩⟩) (.authority (.programFamilyFact))

def exact16979RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], []⟩, (1)⟩]

theorem exact16979RawTermsValid :
    exact16979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18511⟩⟩) exact16979RawTerms (.finite 18) 16978 .exactZero (none)

def event16980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18513⟩⟩) 0 ⟨6544⟩ 16752

def event16981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18513⟩⟩) 1 ⟨18511⟩ 16979

def event16982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18513⟩⟩) (.product (.predecessor 0 16980 .coefficient) (.predecessor 1 16981 .coefficient) (⟨false, true, none, none, some 1⟩))

def event16983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18513⟩⟩, .operator (⟨16752, 0⟩, ⟨16979, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact16984RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact16984RawTermsValid :
    exact16984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18513⟩⟩) exact16984RawTerms .large 16982 .exactZero (none)

def event16985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6744⟩⟩) 0 ⟨6689⟩ 16734

def event16986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6744⟩⟩) (.authority (.operator))

def exact16987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩]

theorem exact16987RawTermsValid :
    exact16987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16987 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6744⟩⟩) exact16987RawTerms .large 16986 .exactZero (none)

def event16988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18514⟩⟩) 0 ⟨6744⟩ 16987

def event16989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18514⟩⟩) 1 ⟨18513⟩ 16984

def event16990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18514⟩⟩) (.sum [.predecessor 0 16988 .coefficient, .predecessor 1 16989 .coefficient])

def exact16991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact16991RawTermsValid :
    exact16991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18514⟩⟩) exact16991RawTerms .large 16990 .exactZero (none)

def event16992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18695⟩⟩) 0 ⟨18514⟩ 16991

def event16993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18695⟩⟩) 1 ⟨18694⟩ 16976

def event16994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18695⟩⟩) (.sum [.predecessor 0 16992 .coefficient, .predecessor 1 16993 .coefficient])

def exact16995RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact16995RawTermsValid :
    exact16995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16995 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18695⟩⟩) exact16995RawTerms .large 16994 .exactZero (none)

def event16996 : Event := .preFoldPolynomial 16995 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact16997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event16997 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨18695⟩⟩) 16996 exact16997RawTerms .large 16994 .exactZero (none)

def event16998 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨18402⟩⟩) ⟨⟨1⟩, ⟨67⟩, ⟨109⟩⟩ ⟨15636, 16998⟩

def event16999 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨18578⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18575⟩⟩]⟩) (1) 0 2 (.universal 16998 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18575⟩⟩]⟩) (none) 16997)

def event17000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 18, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩)

def event17001 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 34, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17002 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 17, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17003 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 30, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17004 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 16, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17005 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 29, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17006 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 15, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17007 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 28, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17008 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 14, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17009 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 35, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17010 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 13, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17011 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 33, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17012 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 12, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17013 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 31, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17014 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 11, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17015 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 27, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17016 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 10, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 36, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17018 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 9, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17019 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 26, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17020 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 8, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17021 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 25, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17022 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 7, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 24, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17024 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 6, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 23, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17026 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 5, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17027 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 22, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 4, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17029 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 32, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17030 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17031 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 21, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17033 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 20, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17035 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 19, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩)

def event17036 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩)

def event17037 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18578⟩⟩, .relation 16999 37, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact17038RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17038RawTermsValid :
    exact17038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18578⟩⟩) exact17038RawTerms .large 15632 (.finite 1811303510016) (some (15634))

def event17039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30211⟩⟩) 0 ⟨18578⟩ 17038

def event17040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30211⟩⟩) 1 ⟨30210⟩ 15622

def event17041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30211⟩⟩) (.sum [.predecessor 0 17039 .coefficient, .predecessor 1 17040 .coefficient])

def event17042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 34⟩, ⟨15622, 33⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18182⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17043 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 17⟩, ⟨15622, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17044 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 30⟩, ⟨15622, 29⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17097⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17045 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 16⟩, ⟨15622, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17046 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 29⟩, ⟨15622, 28⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16810⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17047 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 15⟩, ⟨15622, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 28⟩, ⟨15622, 27⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17049 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 14⟩, ⟨15622, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17050 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 35⟩, ⟨15622, 34⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18217⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17051 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 13⟩, ⟨15622, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17052 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 33⟩, ⟨15622, 32⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17916⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17053 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 12⟩, ⟨15622, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17054 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 31⟩, ⟨15622, 30⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17132⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17055 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 11⟩, ⟨15622, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17056 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 27⟩, ⟨15622, 26⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16320⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17057 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 10⟩, ⟨15622, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17058 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 36⟩, ⟨15622, 35⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17059 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 9⟩, ⟨15622, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17060 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 26⟩, ⟨15622, 25⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16117⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17061 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 8⟩, ⟨15622, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17062 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 25⟩, ⟨15622, 24⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15998⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17063 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 7⟩, ⟨15622, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17064 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 24⟩, ⟨15622, 23⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15879⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17065 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 6⟩, ⟨15622, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17066 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 23⟩, ⟨15622, 22⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17067 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 5⟩, ⟨15622, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 22⟩, ⟨15622, 21⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17069 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 4⟩, ⟨15622, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17070 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 32⟩, ⟨15622, 31⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17071 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 3⟩, ⟨15622, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17072 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 21⟩, ⟨15622, 20⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17073 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 2⟩, ⟨15622, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17074 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 20⟩, ⟨15622, 19⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17075 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 1⟩, ⟨15622, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17076 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 19⟩, ⟨15622, 18⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩, (-1)⟩)

def event17077 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30211⟩⟩, .operator (⟨17038, 0⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩, (1)⟩)

def event17078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30211⟩⟩) (.sum [.result 17038 .summary, .result 15622 .summary])

def exact17079RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17079RawTermsValid :
    exact17079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30211⟩⟩) exact17079RawTerms .large 17041 (.finite 85361036953731455419885957120) (some (17078))

def event17080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30212⟩⟩) 0 ⟨30211⟩ 17079

def event17081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30212⟩⟩) 1 ⟨6652⟩ 5499

def event17082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30212⟩⟩) (.product (.predecessor 0 17080 .coefficient) (.predecessor 1 17081 .coefficient) (⟨false, false, none, none, none⟩))

def event17083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30212⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) [⟨.result 5495 .coefficient, false, none⟩])

def event17084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30212⟩⟩) (.product (.result 17079 .summary) (.transfer 17083) (⟨false, false, none, none, none⟩))

def event17085 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30212⟩⟩, .operator (⟨17079, 0⟩, ⟨5499, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (1)⟩)

def event17086 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30212⟩⟩, .operator (⟨17079, 1⟩, ⟨5499, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (-1)⟩)

def event17087 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30212⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6651⟩⟩) ⟨6597⟩ 5492)

def event17088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30212⟩⟩, .relation 17087 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact17089RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17089RawTermsValid :
    exact17089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30212⟩⟩) exact17089RawTerms .large 17082 (.finite 313276371396785701094268180805713920) (some (17084))

def event17090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24803⟩⟩) 0 ⟨6689⟩ 5477

def event17091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24803⟩⟩) 1 ⟨24802⟩ 6422

def event17092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24803⟩⟩) (.authority (.operator))

def exact17093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24803⟩⟩]⟩, (1)⟩]

theorem exact17093RawTermsValid :
    exact17093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17093 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24803⟩⟩) exact17093RawTerms .large 17092 .exactZero (none)

def event17094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30198⟩⟩) 0 ⟨24803⟩ 17093

def event17095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30198⟩⟩) (.authority (.operator))

def exact17096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩, (1)⟩]

theorem exact17096RawTermsValid :
    exact17096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30198⟩⟩) exact17096RawTerms (.finite 8192) 17095 .exactZero (none)

def event17097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30200⟩⟩) 0 ⟨25780⟩ 6747

def event17098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30200⟩⟩) 1 ⟨30198⟩ 17096

def event17099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30200⟩⟩) (.product (.predecessor 0 17097 .coefficient) (.predecessor 1 17098 .coefficient) (⟨false, false, none, none, none⟩))

def event17100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30200⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) [⟨.result 17096 .coefficient, false, none⟩])

def event17101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30200⟩⟩) (.product (.result 6747 .summary) (.transfer 17100) (⟨false, false, none, none, none⟩))

def event17102 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30200⟩⟩, .operator (⟨6747, 1⟩, ⟨17096, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩, (-1)⟩)

def event17103 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30200⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30198⟩⟩) ⟨24803⟩ 17093)

def event17104 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30200⟩⟩, .relation 17103 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24803⟩⟩]⟩, (-1)⟩)

def event17105 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30200⟩⟩, .operator (⟨6747, 0⟩, ⟨17096, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩, (1)⟩)

def exact17106RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17027⟩⟩], [⟨.program ⟨214⟩, ⟨24803⟩⟩]⟩, (-1)⟩]

theorem exact17106RawTermsValid :
    exact17106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30200⟩⟩) exact17106RawTerms .large 17099 (.finite 1292539133473715126272) (some (17101))

def event17107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22784⟩⟩) 0 ⟨17028⟩ 68

def event17108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22784⟩⟩) (.authority (.relationPreimageSource ⟨64⟩))

def exact17109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22784⟩⟩]⟩, (1)⟩]

theorem exact17109RawTermsValid :
    exact17109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22784⟩⟩) exact17109RawTerms (.finite 136065468) 17108 .exactZero (none)

def event17110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22786⟩⟩) 0 ⟨22784⟩ 17109

def event17111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22786⟩⟩) 1 ⟨2348⟩ 4

def event17112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22786⟩⟩) (.scale (.predecessor 0 17110 .coefficient) (.value (.predecessor 1 17111 .coefficient)))

def exact17113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22784⟩⟩]⟩, (1)⟩]

theorem exact17113RawTermsValid :
    exact17113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22786⟩⟩) exact17113RawTerms (.finite 136065468) 17112 .exactZero (none)

def event17114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22787⟩⟩) 0 ⟨5565⟩ 6561

def event17115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22787⟩⟩) 1 ⟨22786⟩ 17113

def event17116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22787⟩⟩) (.product (.predecessor 0 17114 .coefficient) (.predecessor 1 17115 .coefficient) (⟨false, false, none, none, none⟩))

def event17117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22787⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22784⟩⟩]⟩) [⟨.result 17109 .coefficient, false, none⟩])

def event17118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22787⟩⟩) (.product (.result 6561 .summary) (.transfer 17117) (⟨false, false, none, none, none⟩))

def event17119 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22787⟩⟩, .operator (⟨6561, 0⟩, ⟨17113, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22784⟩⟩]⟩, (1)⟩)

def event17120 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22785⟩⟩)

def event17121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event17122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event17123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event17124 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event17125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event17126 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event17127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event17128 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event17129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 17128

def event17130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 17126

def event17131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 17129 .coefficient) (.value (.predecessor 1 17130 .coefficient)))

def event17132 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event17133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 17132

def event17134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 17124

def event17135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 17133 .coefficient, .predecessor 1 17134 .coefficient])

def event17136 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event17137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 17136

def event17138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 17122

def event17139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 17138 .coefficient))

def event17140 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event17141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13382⟩⟩) 0 ⟨5560⟩ 17140

def event17142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13382⟩⟩) (.authority (.programFamilyFact))

def exact17143RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13382⟩⟩], []⟩, (1)⟩]

theorem exact17143RawTermsValid :
    exact17143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13382⟩⟩) exact17143RawTerms (.finite 60) 17142 .exactZero (none)

def event17144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10365⟩⟩) 0 ⟨5560⟩ 17140

def event17145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10365⟩⟩) (.authority (.programFamilyFact))

def exact17146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩], []⟩, (1)⟩]

theorem exact17146RawTermsValid :
    exact17146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10365⟩⟩) exact17146RawTerms (.finite 60) 17145 .exactZero (none)

def event17147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13383⟩⟩) 0 ⟨10365⟩ 17146

def event17148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13383⟩⟩) 1 ⟨13382⟩ 17143

def event17149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13383⟩⟩) (.product (.predecessor 0 17147 .coefficient) (.predecessor 1 17148 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13383⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10365⟩⟩, ⟨.program ⟨214⟩, ⟨13382⟩⟩], []⟩) [⟨.result 17146 .coefficient, true, some 1⟩, ⟨.result 17143 .coefficient, true, some 1⟩])

def event17151 : Event := .survivorFold (1) 17150

def eventLeaf1056 : Array AnnotatedEvent := #[
  { event := event16896
    frameStart := 16225 },
  { event := event16897
    frameStart := 16225 },
  { event := event16898
    frameStart := 16225 },
  { event := event16899
    frameStart := 16225 },
  { event := event16900
    frameStart := 16225 },
  { event := event16901
    frameStart := 16225 },
  { event := event16902
    frameStart := 16225 },
  { event := event16903
    frameStart := 16225 },
  { event := event16904
    frameStart := 16225 },
  { event := event16905
    frameStart := 16225 },
  { event := event16906
    frameStart := 16225 },
  { event := event16907
    frameStart := 16225 },
  { event := event16908
    frameStart := 16225 },
  { event := event16909
    frameStart := 16225 },
  { event := event16910
    frameStart := 16225 },
  { event := event16911
    frameStart := 16225 }
]

def eventLeaf1057 : Array AnnotatedEvent := #[
  { event := event16912
    frameStart := 16225 },
  { event := event16913
    frameStart := 16225 },
  { event := event16914
    frameStart := 16225 },
  { event := event16915
    frameStart := 16225 },
  { event := event16916
    frameStart := 16225 },
  { event := event16917
    frameStart := 16225 },
  { event := event16918
    frameStart := 16225 },
  { event := event16919
    frameStart := 16225 },
  { event := event16920
    frameStart := 16225 },
  { event := event16921
    frameStart := 16225 },
  { event := event16922
    frameStart := 16225 },
  { event := event16923
    frameStart := 16225 },
  { event := event16924
    frameStart := 16225 },
  { event := event16925
    frameStart := 16225 },
  { event := event16926
    frameStart := 16225 },
  { event := event16927
    frameStart := 16225 }
]

def eventLeaf1058 : Array AnnotatedEvent := #[
  { event := event16928
    frameStart := 16225 },
  { event := event16929
    frameStart := 16225 },
  { event := event16930
    frameStart := 16225 },
  { event := event16931
    frameStart := 16225 },
  { event := event16932
    frameStart := 16225 },
  { event := event16933
    frameStart := 16225 },
  { event := event16934
    frameStart := 16225 },
  { event := event16935
    frameStart := 16225 },
  { event := event16936
    frameStart := 16225 },
  { event := event16937
    frameStart := 16225 },
  { event := event16938
    frameStart := 16225 },
  { event := event16939
    frameStart := 16225 },
  { event := event16940
    frameStart := 16225 },
  { event := event16941
    frameStart := 16225 },
  { event := event16942
    frameStart := 16225 },
  { event := event16943
    frameStart := 16225 }
]

def eventLeaf1059 : Array AnnotatedEvent := #[
  { event := event16944
    frameStart := 16225 },
  { event := event16945
    frameStart := 16225 },
  { event := event16946
    frameStart := 16225 },
  { event := event16947
    frameStart := 16225 },
  { event := event16948
    frameStart := 16225 },
  { event := event16949
    frameStart := 16225 },
  { event := event16950
    frameStart := 16225 },
  { event := event16951
    frameStart := 16225 },
  { event := event16952
    frameStart := 16225 },
  { event := event16953
    frameStart := 16225 },
  { event := event16954
    frameStart := 16225 },
  { event := event16955
    frameStart := 16225 },
  { event := event16956
    frameStart := 16225 },
  { event := event16957
    frameStart := 16225 },
  { event := event16958
    frameStart := 16225 },
  { event := event16959
    frameStart := 16225 }
]

def eventLeaf1060 : Array AnnotatedEvent := #[
  { event := event16960
    frameStart := 16225 },
  { event := event16961
    frameStart := 16225 },
  { event := event16962
    frameStart := 16225 },
  { event := event16963
    frameStart := 16225 },
  { event := event16964
    frameStart := 16225 },
  { event := event16965
    frameStart := 16225 },
  { event := event16966
    frameStart := 16225 },
  { event := event16967
    frameStart := 16225 },
  { event := event16968
    frameStart := 16225 },
  { event := event16969
    frameStart := 16225 },
  { event := event16970
    frameStart := 16225 },
  { event := event16971
    frameStart := 16225 },
  { event := event16972
    frameStart := 16225 },
  { event := event16973
    frameStart := 16225 },
  { event := event16974
    frameStart := 16225 },
  { event := event16975
    frameStart := 16225 }
]

def eventLeaf1061 : Array AnnotatedEvent := #[
  { event := event16976
    frameStart := 16225 },
  { event := event16977
    frameStart := 16225 },
  { event := event16978
    frameStart := 16225 },
  { event := event16979
    frameStart := 16225 },
  { event := event16980
    frameStart := 16225 },
  { event := event16981
    frameStart := 16225 },
  { event := event16982
    frameStart := 16225 },
  { event := event16983
    frameStart := 16225 },
  { event := event16984
    frameStart := 16225 },
  { event := event16985
    frameStart := 16225 },
  { event := event16986
    frameStart := 16225 },
  { event := event16987
    frameStart := 16225 },
  { event := event16988
    frameStart := 16225 },
  { event := event16989
    frameStart := 16225 },
  { event := event16990
    frameStart := 16225 },
  { event := event16991
    frameStart := 16225 }
]

def eventLeaf1062 : Array AnnotatedEvent := #[
  { event := event16992
    frameStart := 16225 },
  { event := event16993
    frameStart := 16225 },
  { event := event16994
    frameStart := 16225 },
  { event := event16995
    frameStart := 16225 },
  { event := event16996
    frameStart := 16225 },
  { event := event16997
    frameStart := 16225 },
  { event := event16998
    frameStart := 0 },
  { event := event16999
    frameStart := 0 },
  { event := event17000
    frameStart := 0 },
  { event := event17001
    frameStart := 0 },
  { event := event17002
    frameStart := 0 },
  { event := event17003
    frameStart := 0 },
  { event := event17004
    frameStart := 0 },
  { event := event17005
    frameStart := 0 },
  { event := event17006
    frameStart := 0 },
  { event := event17007
    frameStart := 0 }
]

def eventLeaf1063 : Array AnnotatedEvent := #[
  { event := event17008
    frameStart := 0 },
  { event := event17009
    frameStart := 0 },
  { event := event17010
    frameStart := 0 },
  { event := event17011
    frameStart := 0 },
  { event := event17012
    frameStart := 0 },
  { event := event17013
    frameStart := 0 },
  { event := event17014
    frameStart := 0 },
  { event := event17015
    frameStart := 0 },
  { event := event17016
    frameStart := 0 },
  { event := event17017
    frameStart := 0 },
  { event := event17018
    frameStart := 0 },
  { event := event17019
    frameStart := 0 },
  { event := event17020
    frameStart := 0 },
  { event := event17021
    frameStart := 0 },
  { event := event17022
    frameStart := 0 },
  { event := event17023
    frameStart := 0 }
]

def eventLeaf1064 : Array AnnotatedEvent := #[
  { event := event17024
    frameStart := 0 },
  { event := event17025
    frameStart := 0 },
  { event := event17026
    frameStart := 0 },
  { event := event17027
    frameStart := 0 },
  { event := event17028
    frameStart := 0 },
  { event := event17029
    frameStart := 0 },
  { event := event17030
    frameStart := 0 },
  { event := event17031
    frameStart := 0 },
  { event := event17032
    frameStart := 0 },
  { event := event17033
    frameStart := 0 },
  { event := event17034
    frameStart := 0 },
  { event := event17035
    frameStart := 0 },
  { event := event17036
    frameStart := 0 },
  { event := event17037
    frameStart := 0 },
  { event := event17038
    frameStart := 0 },
  { event := event17039
    frameStart := 0 }
]

def eventLeaf1065 : Array AnnotatedEvent := #[
  { event := event17040
    frameStart := 0 },
  { event := event17041
    frameStart := 0 },
  { event := event17042
    frameStart := 0 },
  { event := event17043
    frameStart := 0 },
  { event := event17044
    frameStart := 0 },
  { event := event17045
    frameStart := 0 },
  { event := event17046
    frameStart := 0 },
  { event := event17047
    frameStart := 0 },
  { event := event17048
    frameStart := 0 },
  { event := event17049
    frameStart := 0 },
  { event := event17050
    frameStart := 0 },
  { event := event17051
    frameStart := 0 },
  { event := event17052
    frameStart := 0 },
  { event := event17053
    frameStart := 0 },
  { event := event17054
    frameStart := 0 },
  { event := event17055
    frameStart := 0 }
]

def eventLeaf1066 : Array AnnotatedEvent := #[
  { event := event17056
    frameStart := 0 },
  { event := event17057
    frameStart := 0 },
  { event := event17058
    frameStart := 0 },
  { event := event17059
    frameStart := 0 },
  { event := event17060
    frameStart := 0 },
  { event := event17061
    frameStart := 0 },
  { event := event17062
    frameStart := 0 },
  { event := event17063
    frameStart := 0 },
  { event := event17064
    frameStart := 0 },
  { event := event17065
    frameStart := 0 },
  { event := event17066
    frameStart := 0 },
  { event := event17067
    frameStart := 0 },
  { event := event17068
    frameStart := 0 },
  { event := event17069
    frameStart := 0 },
  { event := event17070
    frameStart := 0 },
  { event := event17071
    frameStart := 0 }
]

def eventLeaf1067 : Array AnnotatedEvent := #[
  { event := event17072
    frameStart := 0 },
  { event := event17073
    frameStart := 0 },
  { event := event17074
    frameStart := 0 },
  { event := event17075
    frameStart := 0 },
  { event := event17076
    frameStart := 0 },
  { event := event17077
    frameStart := 0 },
  { event := event17078
    frameStart := 0 },
  { event := event17079
    frameStart := 0 },
  { event := event17080
    frameStart := 0 },
  { event := event17081
    frameStart := 0 },
  { event := event17082
    frameStart := 0 },
  { event := event17083
    frameStart := 0 },
  { event := event17084
    frameStart := 0 },
  { event := event17085
    frameStart := 0 },
  { event := event17086
    frameStart := 0 },
  { event := event17087
    frameStart := 0 }
]

def eventLeaf1068 : Array AnnotatedEvent := #[
  { event := event17088
    frameStart := 0 },
  { event := event17089
    frameStart := 0 },
  { event := event17090
    frameStart := 0 },
  { event := event17091
    frameStart := 0 },
  { event := event17092
    frameStart := 0 },
  { event := event17093
    frameStart := 0 },
  { event := event17094
    frameStart := 0 },
  { event := event17095
    frameStart := 0 },
  { event := event17096
    frameStart := 0 },
  { event := event17097
    frameStart := 0 },
  { event := event17098
    frameStart := 0 },
  { event := event17099
    frameStart := 0 },
  { event := event17100
    frameStart := 0 },
  { event := event17101
    frameStart := 0 },
  { event := event17102
    frameStart := 0 },
  { event := event17103
    frameStart := 0 }
]

def eventLeaf1069 : Array AnnotatedEvent := #[
  { event := event17104
    frameStart := 0 },
  { event := event17105
    frameStart := 0 },
  { event := event17106
    frameStart := 0 },
  { event := event17107
    frameStart := 0 },
  { event := event17108
    frameStart := 0 },
  { event := event17109
    frameStart := 0 },
  { event := event17110
    frameStart := 0 },
  { event := event17111
    frameStart := 0 },
  { event := event17112
    frameStart := 0 },
  { event := event17113
    frameStart := 0 },
  { event := event17114
    frameStart := 0 },
  { event := event17115
    frameStart := 0 },
  { event := event17116
    frameStart := 0 },
  { event := event17117
    frameStart := 0 },
  { event := event17118
    frameStart := 0 },
  { event := event17119
    frameStart := 0 }
]

def eventLeaf1070 : Array AnnotatedEvent := #[
  { event := event17120
    frameStart := 17120 },
  { event := event17121
    frameStart := 17120 },
  { event := event17122
    frameStart := 17120 },
  { event := event17123
    frameStart := 17120 },
  { event := event17124
    frameStart := 17120 },
  { event := event17125
    frameStart := 17120 },
  { event := event17126
    frameStart := 17120 },
  { event := event17127
    frameStart := 17120 },
  { event := event17128
    frameStart := 17120 },
  { event := event17129
    frameStart := 17120 },
  { event := event17130
    frameStart := 17120 },
  { event := event17131
    frameStart := 17120 },
  { event := event17132
    frameStart := 17120 },
  { event := event17133
    frameStart := 17120 },
  { event := event17134
    frameStart := 17120 },
  { event := event17135
    frameStart := 17120 }
]

def eventLeaf1071 : Array AnnotatedEvent := #[
  { event := event17136
    frameStart := 17120 },
  { event := event17137
    frameStart := 17120 },
  { event := event17138
    frameStart := 17120 },
  { event := event17139
    frameStart := 17120 },
  { event := event17140
    frameStart := 17120 },
  { event := event17141
    frameStart := 17120 },
  { event := event17142
    frameStart := 17120 },
  { event := event17143
    frameStart := 17120 },
  { event := event17144
    frameStart := 17120 },
  { event := event17145
    frameStart := 17120 },
  { event := event17146
    frameStart := 17120 },
  { event := event17147
    frameStart := 17120 },
  { event := event17148
    frameStart := 17120 },
  { event := event17149
    frameStart := 17120 },
  { event := event17150
    frameStart := 17120 },
  { event := event17151
    frameStart := 17120 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events066
