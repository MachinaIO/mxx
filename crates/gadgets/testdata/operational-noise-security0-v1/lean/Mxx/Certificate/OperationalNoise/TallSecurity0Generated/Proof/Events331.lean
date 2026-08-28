import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events331

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact84736RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84736RawTermsValid :
    exact84736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11470⟩⟩) exact84736RawTerms .large 84734 .exactZero (none)

def event84737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7235⟩⟩) 0 ⟨5539⟩ 79790

def event84738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7235⟩⟩) 1 ⟨6779⟩ 11482

def event84739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7235⟩⟩) (.product (.predecessor 0 84737 .coefficient) (.predecessor 1 84738 .coefficient) (⟨false, false, none, none, none⟩))

def event84740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7235⟩⟩, .operator (⟨79790, 0⟩, ⟨11482, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact84741RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact84741RawTermsValid :
    exact84741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7235⟩⟩) exact84741RawTerms .large 84739 .exactZero (none)

def event84742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11471⟩⟩) 0 ⟨7235⟩ 84741

def event84743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11471⟩⟩) 1 ⟨11470⟩ 84736

def event84744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11471⟩⟩) (.sum [.predecessor 0 84742 .coefficient, .predecessor 1 84743 .coefficient])

def exact84745RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84745RawTermsValid :
    exact84745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84745 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11471⟩⟩) exact84745RawTerms .large 84744 .exactZero (none)

def event84746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11472⟩⟩) 0 ⟨11471⟩ 84745

def event84747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11472⟩⟩) 1 ⟨93⟩ 11474

def event84748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11472⟩⟩) (.sum [.predecessor 0 84746 .coefficient, .predecessor 1 84747 .coefficient])

def event84749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11472⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) [⟨.result 11474 .coefficient, false, none⟩])

def event84750 : Event := .survivorFold (1) 84749

def exact84751RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84751RawTermsValid :
    exact84751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11472⟩⟩) exact84751RawTerms .large 84748 (.finite 26) (some (84749))

def event84752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14210⟩⟩) 0 ⟨11472⟩ 84751

def event84753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14210⟩⟩) 1 ⟨14207⟩ 4061

def event84754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14210⟩⟩) (.product (.predecessor 0 84752 .coefficient) (.predecessor 1 84753 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14210⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩) [⟨.result 4061 .coefficient, true, some 1⟩])

def event84756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14210⟩⟩) (.product (.result 84751 .summary) (.transfer 84755) (⟨false, false, none, none, none⟩))

def event84757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14210⟩⟩, .operator (⟨84751, 1⟩, ⟨4061, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event84758 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14210⟩⟩, .operator (⟨84751, 0⟩, ⟨4061, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact84759RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact84759RawTermsValid :
    exact84759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84759 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14210⟩⟩) exact84759RawTerms .large 84754 (.finite 14976) (some (84756))

def event84760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14211⟩⟩) 0 ⟨14207⟩ 4061

def event84761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14211⟩⟩) 1 ⟨6567⟩ 79920

def event84762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14211⟩⟩) (.tensor (.predecessor 0 84760 .coefficient) (.predecessor 1 84761 .coefficient) true false)

def event84763 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14211⟩⟩, .operator (⟨4061, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84764RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84764RawTermsValid :
    exact84764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14211⟩⟩) exact84764RawTerms .large 84762 .exactZero (none)

def event84765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7215⟩⟩) 0 ⟨5539⟩ 79790

def event84766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7215⟩⟩) 1 ⟨6759⟩ 11523

def event84767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7215⟩⟩) (.product (.predecessor 0 84765 .coefficient) (.predecessor 1 84766 .coefficient) (⟨false, false, none, none, none⟩))

def event84768 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7215⟩⟩, .operator (⟨79790, 0⟩, ⟨11523, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩)

def exact84769RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact84769RawTermsValid :
    exact84769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84769 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7215⟩⟩) exact84769RawTerms .large 84767 .exactZero (none)

def event84770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14212⟩⟩) 0 ⟨7215⟩ 84769

def event84771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14212⟩⟩) 1 ⟨14211⟩ 84764

def event84772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14212⟩⟩) (.sum [.predecessor 0 84770 .coefficient, .predecessor 1 84771 .coefficient])

def exact84773RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84773RawTermsValid :
    exact84773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84773 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14212⟩⟩) exact84773RawTerms .large 84772 .exactZero (none)

def event84774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14213⟩⟩) 0 ⟨14212⟩ 84773

def event84775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14213⟩⟩) 1 ⟨73⟩ 11515

def event84776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14213⟩⟩) (.sum [.predecessor 0 84774 .coefficient, .predecessor 1 84775 .coefficient])

def event84777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14213⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) [⟨.result 11515 .coefficient, false, none⟩])

def event84778 : Event := .survivorFold (1) 84777

def exact84779RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84779RawTermsValid :
    exact84779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14213⟩⟩) exact84779RawTerms .large 84776 (.finite 26) (some (84777))

def event84780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14214⟩⟩) 0 ⟨14213⟩ 84779

def event84781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14214⟩⟩) 1 ⟨7853⟩ 11512

def event84782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14214⟩⟩) (.product (.predecessor 0 84780 .coefficient) (.predecessor 1 84781 .coefficient) (⟨false, false, none, none, none⟩))

def event84783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14214⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) [⟨.result 11508 .coefficient, false, none⟩])

def event84784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14214⟩⟩) (.product (.result 84779 .summary) (.transfer 84783) (⟨false, false, none, none, none⟩))

def event84785 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14214⟩⟩, .operator (⟨84779, 1⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (-1)⟩)

def event84786 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14214⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7852⟩⟩) ⟨6779⟩ 11482)

def event84787 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14214⟩⟩, .relation 84786 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩)

def event84788 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14214⟩⟩, .operator (⟨84779, 0⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact84789RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩]

theorem exact84789RawTermsValid :
    exact84789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84789 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14214⟩⟩) exact84789RawTerms .large 84782 (.finite 95420416) (some (84784))

def event84790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14215⟩⟩) 0 ⟨14214⟩ 84789

def event84791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14215⟩⟩) 1 ⟨14210⟩ 84759

def event84792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14215⟩⟩) (.sum [.predecessor 0 84790 .coefficient, .predecessor 1 84791 .coefficient])

def event84793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14215⟩⟩, .operator (⟨84789, 1⟩, ⟨84759, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def event84794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14215⟩⟩) (.sum [.result 84789 .summary, .result 84759 .summary])

def exact84795RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84795RawTermsValid :
    exact84795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84795 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14215⟩⟩) exact84795RawTerms .large 84792 (.finite 95435392) (some (84794))

def event84796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26067⟩⟩) 0 ⟨14215⟩ 84795

def event84797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26067⟩⟩) 1 ⟨26066⟩ 84731

def event84798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26067⟩⟩) (.product (.predecessor 0 84796 .coefficient) (.predecessor 1 84797 .coefficient) (⟨false, false, none, none, none⟩))

def event84799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26067⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩) [⟨.result 84731 .coefficient, false, none⟩])

def event84800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26067⟩⟩) (.product (.result 84795 .summary) (.transfer 84799) (⟨false, false, none, none, none⟩))

def event84801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26067⟩⟩, .operator (⟨84795, 1⟩, ⟨84731, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩)

def event84802 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26067⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26066⟩⟩) ⟨23584⟩ 84728)

def event84803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26067⟩⟩, .relation 84802 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (-1)⟩)

def event84804 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26067⟩⟩, .operator (⟨84795, 0⟩, ⟨84731, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩)

def exact84805RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (-1)⟩]

theorem exact84805RawTermsValid :
    exact84805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26067⟩⟩) exact84805RawTerms .large 84798 (.finite 350249415606272) (some (84800))

def event84806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19528⟩⟩) 0 ⟨14209⟩ 4069

def event84807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19528⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact84808RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩]

theorem exact84808RawTermsValid :
    exact84808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19528⟩⟩) exact84808RawTerms (.finite 136065468) 84807 .exactZero (none)

def event84809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19530⟩⟩) 0 ⟨19528⟩ 84808

def event84810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19530⟩⟩) 1 ⟨2348⟩ 4

def event84811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19530⟩⟩) (.scale (.predecessor 0 84809 .coefficient) (.value (.predecessor 1 84810 .coefficient)))

def exact84812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩]

theorem exact84812RawTermsValid :
    exact84812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84812 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19530⟩⟩) exact84812RawTerms (.finite 136065468) 84811 .exactZero (none)

def event84813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19531⟩⟩) 0 ⟨5541⟩ 80012

def event84814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19531⟩⟩) 1 ⟨19530⟩ 84812

def event84815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19531⟩⟩) (.product (.predecessor 0 84813 .coefficient) (.predecessor 1 84814 .coefficient) (⟨false, false, none, none, none⟩))

def event84816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19531⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩) [⟨.result 84808 .coefficient, false, none⟩])

def event84817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19531⟩⟩) (.product (.result 80012 .summary) (.transfer 84816) (⟨false, false, none, none, none⟩))

def event84818 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19531⟩⟩, .operator (⟨80012, 0⟩, ⟨84812, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩)

def event84819 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19529⟩⟩)

def event84820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84821 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84823 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84825 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84827 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84828 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84827

def event84829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84825

def event84830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84828 .coefficient) (.value (.predecessor 1 84829 .coefficient)))

def event84831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84831

def event84833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84823

def event84834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84832 .coefficient, .predecessor 1 84833 .coefficient])

def event84835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84835

def event84837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84821

def event84838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84837 .coefficient))

def event84839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11469⟩⟩) 0 ⟨5536⟩ 84839

def event84841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11469⟩⟩) (.authority (.programFamilyFact))

def exact84842RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩], []⟩, (1)⟩]

theorem exact84842RawTermsValid :
    exact84842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11469⟩⟩) exact84842RawTerms (.finite 18) 84841 .exactZero (none)

def event84843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14207⟩⟩) 0 ⟨5536⟩ 84839

def event84844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14207⟩⟩) (.authority (.programFamilyFact))

def exact84845RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact84845RawTermsValid :
    exact84845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84845 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14207⟩⟩) exact84845RawTerms (.finite 18) 84844 .exactZero (none)

def event84846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 0 ⟨14207⟩ 84845

def event84847 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 1 ⟨11469⟩ 84842

def event84848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.product (.predecessor 0 84846 .coefficient) (.predecessor 1 84847 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩) [⟨.result 84845 .coefficient, true, some 1⟩, ⟨.result 84842 .coefficient, true, some 1⟩])

def event84850 : Event := .survivorFold (1) 84849

def exact84851RawTerms : List Term := []

theorem exact84851RawTermsValid :
    exact84851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84851 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14208⟩⟩) exact84851RawTerms (.finite 324) 84848 (.finite 324) (some (84849))

def event84852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14209⟩⟩) 0 ⟨14208⟩ 84851

def event84853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.identity (.predecessor 0 84852 .coefficient))

def event84854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.finite 324)

def event84855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19528⟩⟩) 0 ⟨14209⟩ 84854

def event84856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19528⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact84857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩]

theorem exact84857RawTermsValid :
    exact84857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19528⟩⟩) exact84857RawTerms (.finite 136065468) 84856 .exactZero (none)

def event84858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact84859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact84859RawTermsValid :
    exact84859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact84859RawTerms .large 84858 .exactZero (none)

def event84860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19529⟩⟩) 0 ⟨6⟩ 84859

def event84861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19529⟩⟩) 1 ⟨19528⟩ 84857

def event84862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19529⟩⟩) (.product (.predecessor 0 84860 .coefficient) (.predecessor 1 84861 .coefficient) (⟨false, false, none, none, none⟩))

def event84863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19529⟩⟩, .operator (⟨84859, 0⟩, ⟨84857, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩)

def exact84864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩]

theorem exact84864RawTermsValid :
    exact84864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19529⟩⟩) exact84864RawTerms .large 84862 .exactZero (none)

def event84865 : Event := .preFoldPolynomial 84864 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩] .exactZero none

def exact84866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩, (1)⟩]

def event84866 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19529⟩⟩) 84865 exact84866RawTerms .large 84862 .exactZero (none)

def event84867 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26070⟩⟩)

def event84868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event84869 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event84870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event84871 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event84872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event84873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event84874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event84875 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event84876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 84875

def event84877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 84873

def event84878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 84876 .coefficient) (.value (.predecessor 1 84877 .coefficient)))

def event84879 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event84880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 84879

def event84881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 84871

def event84882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 84880 .coefficient, .predecessor 1 84881 .coefficient])

def event84883 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event84884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 84883

def event84885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 84869

def event84886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 84885 .coefficient))

def event84887 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event84888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11469⟩⟩) 0 ⟨5536⟩ 84887

def event84889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11469⟩⟩) (.authority (.programFamilyFact))

def exact84890RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩], []⟩, (1)⟩]

theorem exact84890RawTermsValid :
    exact84890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84890 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11469⟩⟩) exact84890RawTerms (.finite 18) 84889 .exactZero (none)

def event84891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14207⟩⟩) 0 ⟨5536⟩ 84887

def event84892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14207⟩⟩) (.authority (.programFamilyFact))

def exact84893RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact84893RawTermsValid :
    exact84893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14207⟩⟩) exact84893RawTerms (.finite 18) 84892 .exactZero (none)

def event84894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 0 ⟨14207⟩ 84893

def event84895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14208⟩⟩) 1 ⟨11469⟩ 84890

def event84896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14208⟩⟩) (.product (.predecessor 0 84894 .coefficient) (.predecessor 1 84895 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84897 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14208⟩⟩, .operator (⟨84893, 0⟩, ⟨84890, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩)

def exact84898RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact84898RawTermsValid :
    exact84898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84898 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14208⟩⟩) exact84898RawTerms (.finite 324) 84896 .exactZero (none)

def event84899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14209⟩⟩) 0 ⟨14208⟩ 84898

def event84900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.identity (.predecessor 0 84899 .coefficient))

def event84901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14209⟩⟩) (.finite 324)

def event84902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23583⟩⟩) 0 ⟨14209⟩ 84901

def event84903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23583⟩⟩) (.authority (.programFamilyFact))

def event84904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23583⟩⟩) (.finite 3720)

def event84905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event84906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23584⟩⟩) 0 ⟨6689⟩ 84905

def event84907 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23584⟩⟩) 1 ⟨23583⟩ 84904

def event84908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23584⟩⟩) (.authority (.operator))

def exact84909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩]

theorem exact84909RawTermsValid :
    exact84909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84909 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23584⟩⟩) exact84909RawTerms .large 84908 .exactZero (none)

def event84910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26066⟩⟩) 0 ⟨23584⟩ 84909

def event84911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26066⟩⟩) (.authority (.operator))

def exact84912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩]

theorem exact84912RawTermsValid :
    exact84912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84912 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26066⟩⟩) exact84912RawTerms (.finite 8192) 84911 .exactZero (none)

def event84913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event84914 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event84915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14314⟩⟩) 0 ⟨14209⟩ 84901

def event84916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14314⟩⟩) 1 ⟨110⟩ 84914

def event84917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14314⟩⟩) (.sum [.predecessor 0 84915 .coefficient, .predecessor 1 84916 .coefficient])

def event84918 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14314⟩⟩) (.finite 324)

def event84919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14315⟩⟩) 0 ⟨14314⟩ 84918

def event84920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14315⟩⟩) (.identity (.predecessor 0 84919 .coefficient))

def exact84921RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], []⟩, (1)⟩]

theorem exact84921RawTermsValid :
    exact84921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14315⟩⟩) exact84921RawTerms (.finite 324) 84920 .exactZero (none)

def event84922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact84923RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84923RawTermsValid :
    exact84923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact84923RawTerms .large 84922 .exactZero (none)

def event84924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14316⟩⟩) 0 ⟨6544⟩ 84923

def event84925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14316⟩⟩) 1 ⟨14315⟩ 84921

def event84926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14316⟩⟩) (.product (.predecessor 0 84924 .coefficient) (.predecessor 1 84925 .coefficient) (⟨false, false, none, none, none⟩))

def event84927 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14316⟩⟩, .operator (⟨84923, 0⟩, ⟨84921, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84928RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84928RawTermsValid :
    exact84928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14316⟩⟩) exact84928RawTerms .large 84926 .exactZero (none)

def event84929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 84905

def event84930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact84931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact84931RawTermsValid :
    exact84931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact84931RawTerms .large 84930 .exactZero (none)

def event84932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6779⟩⟩) 0 ⟨6757⟩ 84931

def event84933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6779⟩⟩) (.identity (.predecessor 0 84932 .coefficient))

def exact84934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact84934RawTermsValid :
    exact84934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6779⟩⟩) exact84934RawTerms .large 84933 .exactZero (none)

def event84935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7852⟩⟩) 0 ⟨6779⟩ 84934

def event84936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7852⟩⟩) (.authority (.operator))

def exact84937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact84937RawTermsValid :
    exact84937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7852⟩⟩) exact84937RawTerms (.finite 8192) 84936 .exactZero (none)

def event84938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 0 ⟨7852⟩ 84937

def event84939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 1 ⟨2348⟩ 84871

def event84940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7853⟩⟩) (.scale (.predecessor 0 84938 .coefficient) (.value (.predecessor 1 84939 .coefficient)))

def exact84941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact84941RawTermsValid :
    exact84941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7853⟩⟩) exact84941RawTerms (.finite 8192) 84940 .exactZero (none)

def event84942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6759⟩⟩) 0 ⟨6757⟩ 84931

def event84943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6759⟩⟩) (.identity (.predecessor 0 84942 .coefficient))

def exact84944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact84944RawTermsValid :
    exact84944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6759⟩⟩) exact84944RawTerms .large 84943 .exactZero (none)

def event84945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 0 ⟨6759⟩ 84944

def event84946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 1 ⟨7853⟩ 84941

def event84947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7854⟩⟩) (.product (.predecessor 0 84945 .coefficient) (.predecessor 1 84946 .coefficient) (⟨false, false, none, none, none⟩))

def event84948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7854⟩⟩, .operator (⟨84944, 0⟩, ⟨84941, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact84949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact84949RawTermsValid :
    exact84949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7854⟩⟩) exact84949RawTerms .large 84947 .exactZero (none)

def event84950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14317⟩⟩) 0 ⟨7854⟩ 84949

def event84951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14317⟩⟩) 1 ⟨14316⟩ 84928

def event84952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14317⟩⟩) (.sum [.predecessor 0 84950 .coefficient, .predecessor 1 84951 .coefficient])

def exact84953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84953RawTermsValid :
    exact84953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84953 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14317⟩⟩) exact84953RawTerms .large 84952 .exactZero (none)

def event84954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26069⟩⟩) 0 ⟨14317⟩ 84953

def event84955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26069⟩⟩) 1 ⟨26066⟩ 84912

def event84956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26069⟩⟩) (.product (.predecessor 0 84954 .coefficient) (.predecessor 1 84955 .coefficient) (⟨false, false, none, none, none⟩))

def event84957 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26069⟩⟩, .operator (⟨84953, 0⟩, ⟨84912, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩)

def event84958 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26069⟩⟩, .operator (⟨84953, 1⟩, ⟨84912, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩)

def event84959 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26069⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26066⟩⟩) ⟨23584⟩ 84909)

def event84960 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26069⟩⟩, .relation 84959 0, ⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (-1)⟩)

def exact84961RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (-1)⟩]

theorem exact84961RawTermsValid :
    exact84961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26069⟩⟩) exact84961RawTerms .large 84956 .exactZero (none)

def event84962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15940⟩⟩) 0 ⟨14209⟩ 84901

def event84963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15940⟩⟩) (.authority (.programFamilyFact))

def exact84964RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], []⟩, (1)⟩]

theorem exact84964RawTermsValid :
    exact84964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15940⟩⟩) exact84964RawTerms (.finite 18) 84963 .exactZero (none)

def event84965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15942⟩⟩) 0 ⟨6544⟩ 84923

def event84966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15942⟩⟩) 1 ⟨15940⟩ 84964

def event84967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15942⟩⟩) (.product (.predecessor 0 84965 .coefficient) (.predecessor 1 84966 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84968 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15942⟩⟩, .operator (⟨84923, 0⟩, ⟨84964, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact84969RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact84969RawTermsValid :
    exact84969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84969 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15942⟩⟩) exact84969RawTerms .large 84967 .exactZero (none)

def event84970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 84905

def event84971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact84972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact84972RawTermsValid :
    exact84972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact84972RawTerms .large 84971 .exactZero (none)

def event84973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15943⟩⟩) 0 ⟨6697⟩ 84972

def event84974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15943⟩⟩) 1 ⟨15942⟩ 84969

def event84975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15943⟩⟩) (.sum [.predecessor 0 84973 .coefficient, .predecessor 1 84974 .coefficient])

def exact84976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84976RawTermsValid :
    exact84976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15943⟩⟩) exact84976RawTerms .large 84975 .exactZero (none)

def event84977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26070⟩⟩) 0 ⟨15943⟩ 84976

def event84978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26070⟩⟩) 1 ⟨26069⟩ 84961

def event84979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26070⟩⟩) (.sum [.predecessor 0 84977 .coefficient, .predecessor 1 84978 .coefficient])

def exact84980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84980RawTermsValid :
    exact84980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26070⟩⟩) exact84980RawTerms .large 84979 .exactZero (none)

def event84981 : Event := .preFoldPolynomial 84980 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event84982 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26070⟩⟩) 84981 exact84982RawTerms .large 84979 .exactZero (none)

def event84983 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14209⟩⟩) ⟨⟨110⟩, ⟨15⟩, ⟨109⟩⟩ ⟨84819, 84983⟩

def event84984 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19531⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩) (1) 0 2 (.universal 84983 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩) (none) 84982)

def event84985 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19531⟩⟩, .relation 84984 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩)

def event84986 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19531⟩⟩, .relation 84984 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩)

def event84987 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19531⟩⟩, .relation 84984 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩)

def event84988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19531⟩⟩, .relation 84984 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact84989RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11469⟩⟩, ⟨.program ⟨214⟩, ⟨14207⟩⟩], [⟨.program ⟨214⟩, ⟨23584⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact84989RawTermsValid :
    exact84989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84989 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19531⟩⟩) exact84989RawTerms .large 84815 (.finite 1811303510016) (some (84817))

def event84990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26068⟩⟩) 0 ⟨19531⟩ 84989

def event84991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26068⟩⟩) 1 ⟨26067⟩ 84805

def eventLeaf5296 : Array AnnotatedEvent := #[
  { event := event84736
    frameStart := 0 },
  { event := event84737
    frameStart := 0 },
  { event := event84738
    frameStart := 0 },
  { event := event84739
    frameStart := 0 },
  { event := event84740
    frameStart := 0 },
  { event := event84741
    frameStart := 0 },
  { event := event84742
    frameStart := 0 },
  { event := event84743
    frameStart := 0 },
  { event := event84744
    frameStart := 0 },
  { event := event84745
    frameStart := 0 },
  { event := event84746
    frameStart := 0 },
  { event := event84747
    frameStart := 0 },
  { event := event84748
    frameStart := 0 },
  { event := event84749
    frameStart := 0 },
  { event := event84750
    frameStart := 0 },
  { event := event84751
    frameStart := 0 }
]

def eventLeaf5297 : Array AnnotatedEvent := #[
  { event := event84752
    frameStart := 0 },
  { event := event84753
    frameStart := 0 },
  { event := event84754
    frameStart := 0 },
  { event := event84755
    frameStart := 0 },
  { event := event84756
    frameStart := 0 },
  { event := event84757
    frameStart := 0 },
  { event := event84758
    frameStart := 0 },
  { event := event84759
    frameStart := 0 },
  { event := event84760
    frameStart := 0 },
  { event := event84761
    frameStart := 0 },
  { event := event84762
    frameStart := 0 },
  { event := event84763
    frameStart := 0 },
  { event := event84764
    frameStart := 0 },
  { event := event84765
    frameStart := 0 },
  { event := event84766
    frameStart := 0 },
  { event := event84767
    frameStart := 0 }
]

def eventLeaf5298 : Array AnnotatedEvent := #[
  { event := event84768
    frameStart := 0 },
  { event := event84769
    frameStart := 0 },
  { event := event84770
    frameStart := 0 },
  { event := event84771
    frameStart := 0 },
  { event := event84772
    frameStart := 0 },
  { event := event84773
    frameStart := 0 },
  { event := event84774
    frameStart := 0 },
  { event := event84775
    frameStart := 0 },
  { event := event84776
    frameStart := 0 },
  { event := event84777
    frameStart := 0 },
  { event := event84778
    frameStart := 0 },
  { event := event84779
    frameStart := 0 },
  { event := event84780
    frameStart := 0 },
  { event := event84781
    frameStart := 0 },
  { event := event84782
    frameStart := 0 },
  { event := event84783
    frameStart := 0 }
]

def eventLeaf5299 : Array AnnotatedEvent := #[
  { event := event84784
    frameStart := 0 },
  { event := event84785
    frameStart := 0 },
  { event := event84786
    frameStart := 0 },
  { event := event84787
    frameStart := 0 },
  { event := event84788
    frameStart := 0 },
  { event := event84789
    frameStart := 0 },
  { event := event84790
    frameStart := 0 },
  { event := event84791
    frameStart := 0 },
  { event := event84792
    frameStart := 0 },
  { event := event84793
    frameStart := 0 },
  { event := event84794
    frameStart := 0 },
  { event := event84795
    frameStart := 0 },
  { event := event84796
    frameStart := 0 },
  { event := event84797
    frameStart := 0 },
  { event := event84798
    frameStart := 0 },
  { event := event84799
    frameStart := 0 }
]

def eventLeaf5300 : Array AnnotatedEvent := #[
  { event := event84800
    frameStart := 0 },
  { event := event84801
    frameStart := 0 },
  { event := event84802
    frameStart := 0 },
  { event := event84803
    frameStart := 0 },
  { event := event84804
    frameStart := 0 },
  { event := event84805
    frameStart := 0 },
  { event := event84806
    frameStart := 0 },
  { event := event84807
    frameStart := 0 },
  { event := event84808
    frameStart := 0 },
  { event := event84809
    frameStart := 0 },
  { event := event84810
    frameStart := 0 },
  { event := event84811
    frameStart := 0 },
  { event := event84812
    frameStart := 0 },
  { event := event84813
    frameStart := 0 },
  { event := event84814
    frameStart := 0 },
  { event := event84815
    frameStart := 0 }
]

def eventLeaf5301 : Array AnnotatedEvent := #[
  { event := event84816
    frameStart := 0 },
  { event := event84817
    frameStart := 0 },
  { event := event84818
    frameStart := 0 },
  { event := event84819
    frameStart := 84819 },
  { event := event84820
    frameStart := 84819 },
  { event := event84821
    frameStart := 84819 },
  { event := event84822
    frameStart := 84819 },
  { event := event84823
    frameStart := 84819 },
  { event := event84824
    frameStart := 84819 },
  { event := event84825
    frameStart := 84819 },
  { event := event84826
    frameStart := 84819 },
  { event := event84827
    frameStart := 84819 },
  { event := event84828
    frameStart := 84819 },
  { event := event84829
    frameStart := 84819 },
  { event := event84830
    frameStart := 84819 },
  { event := event84831
    frameStart := 84819 }
]

def eventLeaf5302 : Array AnnotatedEvent := #[
  { event := event84832
    frameStart := 84819 },
  { event := event84833
    frameStart := 84819 },
  { event := event84834
    frameStart := 84819 },
  { event := event84835
    frameStart := 84819 },
  { event := event84836
    frameStart := 84819 },
  { event := event84837
    frameStart := 84819 },
  { event := event84838
    frameStart := 84819 },
  { event := event84839
    frameStart := 84819 },
  { event := event84840
    frameStart := 84819 },
  { event := event84841
    frameStart := 84819 },
  { event := event84842
    frameStart := 84819 },
  { event := event84843
    frameStart := 84819 },
  { event := event84844
    frameStart := 84819 },
  { event := event84845
    frameStart := 84819 },
  { event := event84846
    frameStart := 84819 },
  { event := event84847
    frameStart := 84819 }
]

def eventLeaf5303 : Array AnnotatedEvent := #[
  { event := event84848
    frameStart := 84819 },
  { event := event84849
    frameStart := 84819 },
  { event := event84850
    frameStart := 84819 },
  { event := event84851
    frameStart := 84819 },
  { event := event84852
    frameStart := 84819 },
  { event := event84853
    frameStart := 84819 },
  { event := event84854
    frameStart := 84819 },
  { event := event84855
    frameStart := 84819 },
  { event := event84856
    frameStart := 84819 },
  { event := event84857
    frameStart := 84819 },
  { event := event84858
    frameStart := 84819 },
  { event := event84859
    frameStart := 84819 },
  { event := event84860
    frameStart := 84819 },
  { event := event84861
    frameStart := 84819 },
  { event := event84862
    frameStart := 84819 },
  { event := event84863
    frameStart := 84819 }
]

def eventLeaf5304 : Array AnnotatedEvent := #[
  { event := event84864
    frameStart := 84819 },
  { event := event84865
    frameStart := 84819 },
  { event := event84866
    frameStart := 84819 },
  { event := event84867
    frameStart := 84867 },
  { event := event84868
    frameStart := 84867 },
  { event := event84869
    frameStart := 84867 },
  { event := event84870
    frameStart := 84867 },
  { event := event84871
    frameStart := 84867 },
  { event := event84872
    frameStart := 84867 },
  { event := event84873
    frameStart := 84867 },
  { event := event84874
    frameStart := 84867 },
  { event := event84875
    frameStart := 84867 },
  { event := event84876
    frameStart := 84867 },
  { event := event84877
    frameStart := 84867 },
  { event := event84878
    frameStart := 84867 },
  { event := event84879
    frameStart := 84867 }
]

def eventLeaf5305 : Array AnnotatedEvent := #[
  { event := event84880
    frameStart := 84867 },
  { event := event84881
    frameStart := 84867 },
  { event := event84882
    frameStart := 84867 },
  { event := event84883
    frameStart := 84867 },
  { event := event84884
    frameStart := 84867 },
  { event := event84885
    frameStart := 84867 },
  { event := event84886
    frameStart := 84867 },
  { event := event84887
    frameStart := 84867 },
  { event := event84888
    frameStart := 84867 },
  { event := event84889
    frameStart := 84867 },
  { event := event84890
    frameStart := 84867 },
  { event := event84891
    frameStart := 84867 },
  { event := event84892
    frameStart := 84867 },
  { event := event84893
    frameStart := 84867 },
  { event := event84894
    frameStart := 84867 },
  { event := event84895
    frameStart := 84867 }
]

def eventLeaf5306 : Array AnnotatedEvent := #[
  { event := event84896
    frameStart := 84867 },
  { event := event84897
    frameStart := 84867 },
  { event := event84898
    frameStart := 84867 },
  { event := event84899
    frameStart := 84867 },
  { event := event84900
    frameStart := 84867 },
  { event := event84901
    frameStart := 84867 },
  { event := event84902
    frameStart := 84867 },
  { event := event84903
    frameStart := 84867 },
  { event := event84904
    frameStart := 84867 },
  { event := event84905
    frameStart := 84867 },
  { event := event84906
    frameStart := 84867 },
  { event := event84907
    frameStart := 84867 },
  { event := event84908
    frameStart := 84867 },
  { event := event84909
    frameStart := 84867 },
  { event := event84910
    frameStart := 84867 },
  { event := event84911
    frameStart := 84867 }
]

def eventLeaf5307 : Array AnnotatedEvent := #[
  { event := event84912
    frameStart := 84867 },
  { event := event84913
    frameStart := 84867 },
  { event := event84914
    frameStart := 84867 },
  { event := event84915
    frameStart := 84867 },
  { event := event84916
    frameStart := 84867 },
  { event := event84917
    frameStart := 84867 },
  { event := event84918
    frameStart := 84867 },
  { event := event84919
    frameStart := 84867 },
  { event := event84920
    frameStart := 84867 },
  { event := event84921
    frameStart := 84867 },
  { event := event84922
    frameStart := 84867 },
  { event := event84923
    frameStart := 84867 },
  { event := event84924
    frameStart := 84867 },
  { event := event84925
    frameStart := 84867 },
  { event := event84926
    frameStart := 84867 },
  { event := event84927
    frameStart := 84867 }
]

def eventLeaf5308 : Array AnnotatedEvent := #[
  { event := event84928
    frameStart := 84867 },
  { event := event84929
    frameStart := 84867 },
  { event := event84930
    frameStart := 84867 },
  { event := event84931
    frameStart := 84867 },
  { event := event84932
    frameStart := 84867 },
  { event := event84933
    frameStart := 84867 },
  { event := event84934
    frameStart := 84867 },
  { event := event84935
    frameStart := 84867 },
  { event := event84936
    frameStart := 84867 },
  { event := event84937
    frameStart := 84867 },
  { event := event84938
    frameStart := 84867 },
  { event := event84939
    frameStart := 84867 },
  { event := event84940
    frameStart := 84867 },
  { event := event84941
    frameStart := 84867 },
  { event := event84942
    frameStart := 84867 },
  { event := event84943
    frameStart := 84867 }
]

def eventLeaf5309 : Array AnnotatedEvent := #[
  { event := event84944
    frameStart := 84867 },
  { event := event84945
    frameStart := 84867 },
  { event := event84946
    frameStart := 84867 },
  { event := event84947
    frameStart := 84867 },
  { event := event84948
    frameStart := 84867 },
  { event := event84949
    frameStart := 84867 },
  { event := event84950
    frameStart := 84867 },
  { event := event84951
    frameStart := 84867 },
  { event := event84952
    frameStart := 84867 },
  { event := event84953
    frameStart := 84867 },
  { event := event84954
    frameStart := 84867 },
  { event := event84955
    frameStart := 84867 },
  { event := event84956
    frameStart := 84867 },
  { event := event84957
    frameStart := 84867 },
  { event := event84958
    frameStart := 84867 },
  { event := event84959
    frameStart := 84867 }
]

def eventLeaf5310 : Array AnnotatedEvent := #[
  { event := event84960
    frameStart := 84867 },
  { event := event84961
    frameStart := 84867 },
  { event := event84962
    frameStart := 84867 },
  { event := event84963
    frameStart := 84867 },
  { event := event84964
    frameStart := 84867 },
  { event := event84965
    frameStart := 84867 },
  { event := event84966
    frameStart := 84867 },
  { event := event84967
    frameStart := 84867 },
  { event := event84968
    frameStart := 84867 },
  { event := event84969
    frameStart := 84867 },
  { event := event84970
    frameStart := 84867 },
  { event := event84971
    frameStart := 84867 },
  { event := event84972
    frameStart := 84867 },
  { event := event84973
    frameStart := 84867 },
  { event := event84974
    frameStart := 84867 },
  { event := event84975
    frameStart := 84867 }
]

def eventLeaf5311 : Array AnnotatedEvent := #[
  { event := event84976
    frameStart := 84867 },
  { event := event84977
    frameStart := 84867 },
  { event := event84978
    frameStart := 84867 },
  { event := event84979
    frameStart := 84867 },
  { event := event84980
    frameStart := 84867 },
  { event := event84981
    frameStart := 84867 },
  { event := event84982
    frameStart := 84867 },
  { event := event84983
    frameStart := 0 },
  { event := event84984
    frameStart := 0 },
  { event := event84985
    frameStart := 0 },
  { event := event84986
    frameStart := 0 },
  { event := event84987
    frameStart := 0 },
  { event := event84988
    frameStart := 0 },
  { event := event84989
    frameStart := 0 },
  { event := event84990
    frameStart := 0 },
  { event := event84991
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events331
