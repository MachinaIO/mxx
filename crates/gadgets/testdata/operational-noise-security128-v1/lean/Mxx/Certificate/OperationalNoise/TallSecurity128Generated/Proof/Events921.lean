import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events921

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event235776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact235777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact235777RawTermsValid :
    exact235777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact235777RawTerms .large 235776 .exactZero (none)

def event235778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23284⟩⟩) 0 ⟨6908⟩ 235777

def event235779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23284⟩⟩) 1 ⟨23283⟩ 235775

def event235780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23284⟩⟩) (.product (.predecessor 0 235778 .coefficient) (.predecessor 1 235779 .coefficient) (⟨false, false, none, none, none⟩))

def event235781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23284⟩⟩, .operator (⟨235777, 0⟩, ⟨235775, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact235782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact235782RawTermsValid :
    exact235782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23284⟩⟩) exact235782RawTerms .large 235780 .exactZero (none)

def event235783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 235759

def event235784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact235785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact235785RawTermsValid :
    exact235785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact235785RawTerms .large 235784 .exactZero (none)

def event235786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23285⟩⟩) 0 ⟨7181⟩ 235785

def event235787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23285⟩⟩) 1 ⟨23284⟩ 235782

def event235788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23285⟩⟩) (.sum [.predecessor 0 235786 .coefficient, .predecessor 1 235787 .coefficient])

def exact235789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235789RawTermsValid :
    exact235789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235789 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23285⟩⟩) exact235789RawTerms .large 235788 .exactZero (none)

def event235790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23835⟩⟩) 0 ⟨23285⟩ 235789

def event235791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23835⟩⟩) 1 ⟨23834⟩ 235766

def event235792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23835⟩⟩) (.product (.predecessor 0 235790 .coefficient) (.predecessor 1 235791 .coefficient) (⟨false, false, none, none, none⟩))

def event235793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23835⟩⟩, .operator (⟨235789, 0⟩, ⟨235766, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (1)⟩)

def event235794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23835⟩⟩, .operator (⟨235789, 1⟩, ⟨235766, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩)

def event235795 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23835⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23834⟩⟩) ⟨23071⟩ 235763)

def event235796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23835⟩⟩, .relation 235795 0, ⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (-1)⟩)

def exact235797RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (-1)⟩]

theorem exact235797RawTermsValid :
    exact235797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23835⟩⟩) exact235797RawTerms .large 235792 .exactZero (none)

def event235798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22062⟩⟩) 0 ⟨21801⟩ 235755

def event235799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22062⟩⟩) (.authority (.programFamilyFact))

def exact235800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], []⟩, (1)⟩]

theorem exact235800RawTermsValid :
    exact235800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22062⟩⟩) exact235800RawTerms (.finite 4) 235799 .exactZero (none)

def event235801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22065⟩⟩) 0 ⟨6908⟩ 235777

def event235802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22065⟩⟩) 1 ⟨22062⟩ 235800

def event235803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22065⟩⟩) (.product (.predecessor 0 235801 .coefficient) (.predecessor 1 235802 .coefficient) (⟨false, true, none, none, some 1⟩))

def event235804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22065⟩⟩, .operator (⟨235777, 0⟩, ⟨235800, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact235805RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact235805RawTermsValid :
    exact235805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22065⟩⟩) exact235805RawTerms .large 235803 .exactZero (none)

def event235806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7201⟩⟩) 0 ⟨7177⟩ 235759

def event235807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7201⟩⟩) (.authority (.operator))

def exact235808RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩]

theorem exact235808RawTermsValid :
    exact235808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7201⟩⟩) exact235808RawTerms .large 235807 .exactZero (none)

def event235809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22066⟩⟩) 0 ⟨7201⟩ 235808

def event235810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22066⟩⟩) 1 ⟨22065⟩ 235805

def event235811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22066⟩⟩) (.sum [.predecessor 0 235809 .coefficient, .predecessor 1 235810 .coefficient])

def exact235812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235812RawTermsValid :
    exact235812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22066⟩⟩) exact235812RawTerms .large 235811 .exactZero (none)

def event235813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23840⟩⟩) 0 ⟨22066⟩ 235812

def event235814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23840⟩⟩) 1 ⟨23835⟩ 235797

def event235815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23840⟩⟩) (.sum [.predecessor 0 235813 .coefficient, .predecessor 1 235814 .coefficient])

def exact235816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235816RawTermsValid :
    exact235816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23840⟩⟩) exact235816RawTerms .large 235815 .exactZero (none)

def event235817 : Event := .preFoldPolynomial 235816 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact235818RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event235818 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨23840⟩⟩) 235817 exact235818RawTerms .large 235815 .exactZero (none)

def event235819 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21801⟩⟩) ⟨⟨80⟩, ⟨60⟩, ⟨135⟩⟩ ⟨235661, 235819⟩

def event235820 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22655⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22652⟩⟩]⟩) (1) 0 2 (.universal 235819 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22652⟩⟩]⟩) (none) 235818)

def event235821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22655⟩⟩, .relation 235820 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩)

def event235822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22655⟩⟩, .relation 235820 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩)

def event235823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22655⟩⟩, .relation 235820 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (1)⟩)

def event235824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22655⟩⟩, .relation 235820 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact235825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235825RawTermsValid :
    exact235825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22655⟩⟩) exact235825RawTerms .large 235657 (.finite 202072841853861888) (some (235659))

def event235826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23837⟩⟩) 0 ⟨22655⟩ 235825

def event235827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23837⟩⟩) 1 ⟨23836⟩ 235647

def event235828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23837⟩⟩) (.sum [.predecessor 0 235826 .coefficient, .predecessor 1 235827 .coefficient])

def event235829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23837⟩⟩, .operator (⟨235825, 0⟩, ⟨235647, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23834⟩⟩]⟩, (1)⟩)

def event235830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23837⟩⟩, .operator (⟨235825, 2⟩, ⟨235647, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨21800⟩⟩], [⟨.program ⟨257⟩, ⟨23071⟩⟩]⟩, (-1)⟩)

def event235831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23837⟩⟩) (.sum [.result 235825 .summary, .result 235647 .summary])

def exact235832RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235832RawTermsValid :
    exact235832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23837⟩⟩) exact235832RawTerms .large 235828 (.finite 32189003662929394266751515230208) (some (235831))

def event235833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23838⟩⟩) 0 ⟨23837⟩ 235832

def event235834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23838⟩⟩) 1 ⟨7156⟩ 15842

def event235835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23838⟩⟩) (.product (.predecessor 0 235833 .coefficient) (.predecessor 1 235834 .coefficient) (⟨false, false, none, none, none⟩))

def event235836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23838⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) [⟨.result 15838 .coefficient, false, none⟩])

def event235837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23838⟩⟩) (.product (.result 235832 .summary) (.transfer 235836) (⟨false, false, none, none, none⟩))

def event235838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23838⟩⟩, .operator (⟨235832, 0⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event235839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23838⟩⟩, .operator (⟨235832, 1⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event235840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23838⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835)

def event235841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23838⟩⟩, .relation 235840 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact235842RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact235842RawTermsValid :
    exact235842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23838⟩⟩) exact235842RawTerms .large 235835 (.finite 345626795057764889831969145180473178193920) (some (235837))

def event235843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19851⟩⟩) 0 ⟨7177⟩ 15500

def event235844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19851⟩⟩) 1 ⟨19850⟩ 229859

def event235845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19851⟩⟩) (.authority (.operator))

def exact235846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (1)⟩]

theorem exact235846RawTermsValid :
    exact235846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19851⟩⟩) exact235846RawTerms .large 235845 .exactZero (none)

def event235847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20614⟩⟩) 0 ⟨19851⟩ 235846

def event235848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20614⟩⟩) (.authority (.operator))

def exact235849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩]

theorem exact235849RawTermsValid :
    exact235849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20614⟩⟩) exact235849RawTerms (.finite 8192) 235848 .exactZero (none)

def event235850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20616⟩⟩) 0 ⟨20210⟩ 230143

def event235851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20616⟩⟩) 1 ⟨20614⟩ 235849

def event235852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20616⟩⟩) (.product (.predecessor 0 235850 .coefficient) (.predecessor 1 235851 .coefficient) (⟨false, false, none, none, none⟩))

def event235853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20616⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) [⟨.result 235849 .coefficient, false, none⟩])

def event235854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20616⟩⟩) (.product (.result 230143 .summary) (.transfer 235853) (⟨false, false, none, none, none⟩))

def event235855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20616⟩⟩, .operator (⟨230143, 0⟩, ⟨235849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩)

def event235856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20616⟩⟩, .operator (⟨230143, 1⟩, ⟨235849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (-1)⟩)

def event235857 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20616⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20614⟩⟩) ⟨19851⟩ 235846)

def event235858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20616⟩⟩, .relation 235857 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (-1)⟩)

def exact235859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (-1)⟩]

theorem exact235859RawTermsValid :
    exact235859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20616⟩⟩) exact235859RawTerms .large 235852 (.finite 32188905437706348505289216491520) (some (235854))

def event235860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19432⟩⟩) 0 ⟨18581⟩ 10951

def event235861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19432⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact235862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩]

theorem exact235862RawTermsValid :
    exact235862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19432⟩⟩) exact235862RawTerms (.finite 5647228698) 235861 .exactZero (none)

def event235863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19434⟩⟩) 0 ⟨19432⟩ 235862

def event235864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19434⟩⟩) 1 ⟨2370⟩ 4

def event235865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19434⟩⟩) (.scale (.predecessor 0 235863 .coefficient) (.value (.predecessor 1 235864 .coefficient)))

def exact235866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩]

theorem exact235866RawTermsValid :
    exact235866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19434⟩⟩) exact235866RawTerms (.finite 5647228698) 235865 .exactZero (none)

def event235867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19435⟩⟩) 0 ⟨5581⟩ 222245

def event235868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19435⟩⟩) 1 ⟨19434⟩ 235866

def event235869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19435⟩⟩) (.product (.predecessor 0 235867 .coefficient) (.predecessor 1 235868 .coefficient) (⟨false, false, none, none, none⟩))

def event235870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19435⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩) [⟨.result 235862 .coefficient, false, none⟩])

def event235871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19435⟩⟩) (.product (.result 222245 .summary) (.transfer 235870) (⟨false, false, none, none, none⟩))

def event235872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19435⟩⟩, .operator (⟨222245, 0⟩, ⟨235866, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩)

def event235873 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19433⟩⟩)

def event235874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event235875 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event235876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event235877 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event235878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event235879 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event235880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event235881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event235882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 235881

def event235883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 235879

def event235884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 235882 .coefficient) (.value (.predecessor 1 235883 .coefficient)))

def event235885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event235886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 235885

def event235887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 235877

def event235888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 235886 .coefficient, .predecessor 1 235887 .coefficient])

def event235889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event235890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 235889

def event235891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 235875

def event235892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 235891 .coefficient))

def event235893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event235894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18250⟩⟩) 0 ⟨5577⟩ 235893

def event235895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18250⟩⟩) (.authority (.programFamilyFact))

def exact235896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩, (1)⟩]

theorem exact235896RawTermsValid :
    exact235896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18250⟩⟩) exact235896RawTerms (.finite 3) 235895 .exactZero (none)

def event235897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12666⟩⟩) 0 ⟨5577⟩ 235893

def event235898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12666⟩⟩) (.authority (.programFamilyFact))

def exact235899RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩], []⟩, (1)⟩]

theorem exact235899RawTermsValid :
    exact235899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235899 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12666⟩⟩) exact235899RawTerms (.finite 3) 235898 .exactZero (none)

def event235900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18251⟩⟩) 0 ⟨12666⟩ 235899

def event235901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18251⟩⟩) 1 ⟨18250⟩ 235896

def event235902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18251⟩⟩) (.product (.predecessor 0 235900 .coefficient) (.predecessor 1 235901 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event235903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18251⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩) [⟨.result 235899 .coefficient, true, some 1⟩, ⟨.result 235896 .coefficient, true, some 1⟩])

def event235904 : Event := .survivorFold (1) 235903

def exact235905RawTerms : List Term := []

theorem exact235905RawTermsValid :
    exact235905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18251⟩⟩) exact235905RawTerms (.finite 9) 235902 (.finite 9) (some (235903))

def event235906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18252⟩⟩) 0 ⟨18251⟩ 235905

def event235907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18252⟩⟩) (.identity (.predecessor 0 235906 .coefficient))

def event235908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18252⟩⟩) (.finite 9)

def event235909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18580⟩⟩) 0 ⟨18252⟩ 235908

def event235910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18580⟩⟩) (.authority (.programFamilyFact))

def exact235911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], []⟩, (1)⟩]

theorem exact235911RawTermsValid :
    exact235911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18580⟩⟩) exact235911RawTerms (.finite 3) 235910 .exactZero (none)

def event235912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18581⟩⟩) 0 ⟨18580⟩ 235911

def event235913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18581⟩⟩) (.identity (.predecessor 0 235912 .coefficient))

def event235914 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18581⟩⟩) (.finite 3)

def event235915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19432⟩⟩) 0 ⟨18581⟩ 235914

def event235916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19432⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact235917RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩]

theorem exact235917RawTermsValid :
    exact235917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19432⟩⟩) exact235917RawTerms (.finite 5647228698) 235916 .exactZero (none)

def event235918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact235919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact235919RawTermsValid :
    exact235919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact235919RawTerms .large 235918 .exactZero (none)

def event235920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19433⟩⟩) 0 ⟨35⟩ 235919

def event235921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19433⟩⟩) 1 ⟨19432⟩ 235917

def event235922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19433⟩⟩) (.product (.predecessor 0 235920 .coefficient) (.predecessor 1 235921 .coefficient) (⟨false, false, none, none, none⟩))

def event235923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19433⟩⟩, .operator (⟨235919, 0⟩, ⟨235917, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩)

def exact235924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩]

theorem exact235924RawTermsValid :
    exact235924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19433⟩⟩) exact235924RawTerms .large 235922 .exactZero (none)

def event235925 : Event := .preFoldPolynomial 235924 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩] .exactZero none

def exact235926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19432⟩⟩]⟩, (1)⟩]

def event235926 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨19433⟩⟩) 235925 exact235926RawTerms .large 235922 .exactZero (none)

def event235927 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨20620⟩⟩)

def event235928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event235929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event235930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event235931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event235932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event235933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event235934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event235935 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event235936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 235935

def event235937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 235933

def event235938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 235936 .coefficient) (.value (.predecessor 1 235937 .coefficient)))

def event235939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event235940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 235939

def event235941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 235931

def event235942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 235940 .coefficient, .predecessor 1 235941 .coefficient])

def event235943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event235944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 235943

def event235945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 235929

def event235946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 235945 .coefficient))

def event235947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event235948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18250⟩⟩) 0 ⟨5577⟩ 235947

def event235949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18250⟩⟩) (.authority (.programFamilyFact))

def exact235950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩, (1)⟩]

theorem exact235950RawTermsValid :
    exact235950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18250⟩⟩) exact235950RawTerms (.finite 3) 235949 .exactZero (none)

def event235951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12666⟩⟩) 0 ⟨5577⟩ 235947

def event235952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12666⟩⟩) (.authority (.programFamilyFact))

def exact235953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩], []⟩, (1)⟩]

theorem exact235953RawTermsValid :
    exact235953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12666⟩⟩) exact235953RawTerms (.finite 3) 235952 .exactZero (none)

def event235954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18251⟩⟩) 0 ⟨12666⟩ 235953

def event235955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18251⟩⟩) 1 ⟨18250⟩ 235950

def event235956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18251⟩⟩) (.product (.predecessor 0 235954 .coefficient) (.predecessor 1 235955 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event235957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18251⟩⟩, .operator (⟨235953, 0⟩, ⟨235950, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩, (1)⟩)

def exact235958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩, (1)⟩]

theorem exact235958RawTermsValid :
    exact235958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18251⟩⟩) exact235958RawTerms (.finite 9) 235956 .exactZero (none)

def event235959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18252⟩⟩) 0 ⟨18251⟩ 235958

def event235960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18252⟩⟩) (.identity (.predecessor 0 235959 .coefficient))

def event235961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18252⟩⟩) (.finite 9)

def event235962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18580⟩⟩) 0 ⟨18252⟩ 235961

def event235963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18580⟩⟩) (.authority (.programFamilyFact))

def exact235964RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], []⟩, (1)⟩]

theorem exact235964RawTermsValid :
    exact235964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18580⟩⟩) exact235964RawTerms (.finite 3) 235963 .exactZero (none)

def event235965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18581⟩⟩) 0 ⟨18580⟩ 235964

def event235966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18581⟩⟩) (.identity (.predecessor 0 235965 .coefficient))

def event235967 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18581⟩⟩) (.finite 3)

def event235968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19850⟩⟩) 0 ⟨18581⟩ 235967

def event235969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19850⟩⟩) (.authority (.programFamilyFact))

def event235970 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19850⟩⟩) (.finite 3720)

def event235971 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event235972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19851⟩⟩) 0 ⟨7177⟩ 235971

def event235973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19851⟩⟩) 1 ⟨19850⟩ 235970

def event235974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19851⟩⟩) (.authority (.operator))

def exact235975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (1)⟩]

theorem exact235975RawTermsValid :
    exact235975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19851⟩⟩) exact235975RawTerms .large 235974 .exactZero (none)

def event235976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20614⟩⟩) 0 ⟨19851⟩ 235975

def event235977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20614⟩⟩) (.authority (.operator))

def exact235978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩]

theorem exact235978RawTermsValid :
    exact235978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20614⟩⟩) exact235978RawTerms (.finite 8192) 235977 .exactZero (none)

def event235979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event235980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event235981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20062⟩⟩) 0 ⟨18581⟩ 235967

def event235982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20062⟩⟩) 1 ⟨136⟩ 235980

def event235983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20062⟩⟩) (.sum [.predecessor 0 235981 .coefficient, .predecessor 1 235982 .coefficient])

def event235984 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨20062⟩⟩) (.finite 3)

def event235985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20063⟩⟩) 0 ⟨20062⟩ 235984

def event235986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20063⟩⟩) (.identity (.predecessor 0 235985 .coefficient))

def exact235987RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], []⟩, (1)⟩]

theorem exact235987RawTermsValid :
    exact235987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20063⟩⟩) exact235987RawTerms (.finite 3) 235986 .exactZero (none)

def event235988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact235989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact235989RawTermsValid :
    exact235989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact235989RawTerms .large 235988 .exactZero (none)

def event235990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20064⟩⟩) 0 ⟨6908⟩ 235989

def event235991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20064⟩⟩) 1 ⟨20063⟩ 235987

def event235992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20064⟩⟩) (.product (.predecessor 0 235990 .coefficient) (.predecessor 1 235991 .coefficient) (⟨false, false, none, none, none⟩))

def event235993 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20064⟩⟩, .operator (⟨235989, 0⟩, ⟨235987, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact235994RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact235994RawTermsValid :
    exact235994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235994 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20064⟩⟩) exact235994RawTerms .large 235992 .exactZero (none)

def event235995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7180⟩⟩) 0 ⟨7177⟩ 235971

def event235996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7180⟩⟩) (.authority (.operator))

def exact235997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩]

theorem exact235997RawTermsValid :
    exact235997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event235997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7180⟩⟩) exact235997RawTerms .large 235996 .exactZero (none)

def event235998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20065⟩⟩) 0 ⟨7180⟩ 235997

def event235999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20065⟩⟩) 1 ⟨20064⟩ 235994

def event236000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20065⟩⟩) (.sum [.predecessor 0 235998 .coefficient, .predecessor 1 235999 .coefficient])

def exact236001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact236001RawTermsValid :
    exact236001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20065⟩⟩) exact236001RawTerms .large 236000 .exactZero (none)

def event236002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20615⟩⟩) 0 ⟨20065⟩ 236001

def event236003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20615⟩⟩) 1 ⟨20614⟩ 235978

def event236004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20615⟩⟩) (.product (.predecessor 0 236002 .coefficient) (.predecessor 1 236003 .coefficient) (⟨false, false, none, none, none⟩))

def event236005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20615⟩⟩, .operator (⟨236001, 0⟩, ⟨235978, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩)

def event236006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20615⟩⟩, .operator (⟨236001, 1⟩, ⟨235978, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (-1)⟩)

def event236007 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20615⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20614⟩⟩) ⟨19851⟩ 235975)

def event236008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20615⟩⟩, .relation 236007 0, ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (-1)⟩)

def exact236009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (-1)⟩]

theorem exact236009RawTermsValid :
    exact236009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236009 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20615⟩⟩) exact236009RawTerms .large 236004 .exactZero (none)

def event236010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18842⟩⟩) 0 ⟨18581⟩ 235967

def event236011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18842⟩⟩) (.authority (.programFamilyFact))

def exact236012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], []⟩, (1)⟩]

theorem exact236012RawTermsValid :
    exact236012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18842⟩⟩) exact236012RawTerms (.finite 3) 236011 .exactZero (none)

def event236013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18845⟩⟩) 0 ⟨6908⟩ 235989

def event236014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18845⟩⟩) 1 ⟨18842⟩ 236012

def event236015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18845⟩⟩) (.product (.predecessor 0 236013 .coefficient) (.predecessor 1 236014 .coefficient) (⟨false, true, none, none, some 1⟩))

def event236016 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18845⟩⟩, .operator (⟨235989, 0⟩, ⟨236012, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact236017RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact236017RawTermsValid :
    exact236017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18845⟩⟩) exact236017RawTerms .large 236015 .exactZero (none)

def event236018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7199⟩⟩) 0 ⟨7177⟩ 235971

def event236019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7199⟩⟩) (.authority (.operator))

def exact236020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩, (1)⟩]

theorem exact236020RawTermsValid :
    exact236020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7199⟩⟩) exact236020RawTerms .large 236019 .exactZero (none)

def event236021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18846⟩⟩) 0 ⟨7199⟩ 236020

def event236022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18846⟩⟩) 1 ⟨18845⟩ 236017

def event236023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18846⟩⟩) (.sum [.predecessor 0 236021 .coefficient, .predecessor 1 236022 .coefficient])

def exact236024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact236024RawTermsValid :
    exact236024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18846⟩⟩) exact236024RawTerms .large 236023 .exactZero (none)

def event236025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20620⟩⟩) 0 ⟨18846⟩ 236024

def event236026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20620⟩⟩) 1 ⟨20615⟩ 236009

def event236027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20620⟩⟩) (.sum [.predecessor 0 236025 .coefficient, .predecessor 1 236026 .coefficient])

def exact236028RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact236028RawTermsValid :
    exact236028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event236028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20620⟩⟩) exact236028RawTerms .large 236027 .exactZero (none)

def event236029 : Event := .preFoldPolynomial 236028 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact236030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20614⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨19851⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event236030 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨20620⟩⟩) 236029 exact236030RawTerms .large 236027 .exactZero (none)

def event236031 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨18581⟩⟩) ⟨⟨78⟩, ⟨58⟩, ⟨135⟩⟩ ⟨235873, 236031⟩

def eventLeaf14736 : Array AnnotatedEvent := #[
  { event := event235776
    frameStart := 235715 },
  { event := event235777
    frameStart := 235715 },
  { event := event235778
    frameStart := 235715 },
  { event := event235779
    frameStart := 235715 },
  { event := event235780
    frameStart := 235715 },
  { event := event235781
    frameStart := 235715 },
  { event := event235782
    frameStart := 235715 },
  { event := event235783
    frameStart := 235715 },
  { event := event235784
    frameStart := 235715 },
  { event := event235785
    frameStart := 235715 },
  { event := event235786
    frameStart := 235715 },
  { event := event235787
    frameStart := 235715 },
  { event := event235788
    frameStart := 235715 },
  { event := event235789
    frameStart := 235715 },
  { event := event235790
    frameStart := 235715 },
  { event := event235791
    frameStart := 235715 }
]

def eventLeaf14737 : Array AnnotatedEvent := #[
  { event := event235792
    frameStart := 235715 },
  { event := event235793
    frameStart := 235715 },
  { event := event235794
    frameStart := 235715 },
  { event := event235795
    frameStart := 235715 },
  { event := event235796
    frameStart := 235715 },
  { event := event235797
    frameStart := 235715 },
  { event := event235798
    frameStart := 235715 },
  { event := event235799
    frameStart := 235715 },
  { event := event235800
    frameStart := 235715 },
  { event := event235801
    frameStart := 235715 },
  { event := event235802
    frameStart := 235715 },
  { event := event235803
    frameStart := 235715 },
  { event := event235804
    frameStart := 235715 },
  { event := event235805
    frameStart := 235715 },
  { event := event235806
    frameStart := 235715 },
  { event := event235807
    frameStart := 235715 }
]

def eventLeaf14738 : Array AnnotatedEvent := #[
  { event := event235808
    frameStart := 235715 },
  { event := event235809
    frameStart := 235715 },
  { event := event235810
    frameStart := 235715 },
  { event := event235811
    frameStart := 235715 },
  { event := event235812
    frameStart := 235715 },
  { event := event235813
    frameStart := 235715 },
  { event := event235814
    frameStart := 235715 },
  { event := event235815
    frameStart := 235715 },
  { event := event235816
    frameStart := 235715 },
  { event := event235817
    frameStart := 235715 },
  { event := event235818
    frameStart := 235715 },
  { event := event235819
    frameStart := 0 },
  { event := event235820
    frameStart := 0 },
  { event := event235821
    frameStart := 0 },
  { event := event235822
    frameStart := 0 },
  { event := event235823
    frameStart := 0 }
]

def eventLeaf14739 : Array AnnotatedEvent := #[
  { event := event235824
    frameStart := 0 },
  { event := event235825
    frameStart := 0 },
  { event := event235826
    frameStart := 0 },
  { event := event235827
    frameStart := 0 },
  { event := event235828
    frameStart := 0 },
  { event := event235829
    frameStart := 0 },
  { event := event235830
    frameStart := 0 },
  { event := event235831
    frameStart := 0 },
  { event := event235832
    frameStart := 0 },
  { event := event235833
    frameStart := 0 },
  { event := event235834
    frameStart := 0 },
  { event := event235835
    frameStart := 0 },
  { event := event235836
    frameStart := 0 },
  { event := event235837
    frameStart := 0 },
  { event := event235838
    frameStart := 0 },
  { event := event235839
    frameStart := 0 }
]

def eventLeaf14740 : Array AnnotatedEvent := #[
  { event := event235840
    frameStart := 0 },
  { event := event235841
    frameStart := 0 },
  { event := event235842
    frameStart := 0 },
  { event := event235843
    frameStart := 0 },
  { event := event235844
    frameStart := 0 },
  { event := event235845
    frameStart := 0 },
  { event := event235846
    frameStart := 0 },
  { event := event235847
    frameStart := 0 },
  { event := event235848
    frameStart := 0 },
  { event := event235849
    frameStart := 0 },
  { event := event235850
    frameStart := 0 },
  { event := event235851
    frameStart := 0 },
  { event := event235852
    frameStart := 0 },
  { event := event235853
    frameStart := 0 },
  { event := event235854
    frameStart := 0 },
  { event := event235855
    frameStart := 0 }
]

def eventLeaf14741 : Array AnnotatedEvent := #[
  { event := event235856
    frameStart := 0 },
  { event := event235857
    frameStart := 0 },
  { event := event235858
    frameStart := 0 },
  { event := event235859
    frameStart := 0 },
  { event := event235860
    frameStart := 0 },
  { event := event235861
    frameStart := 0 },
  { event := event235862
    frameStart := 0 },
  { event := event235863
    frameStart := 0 },
  { event := event235864
    frameStart := 0 },
  { event := event235865
    frameStart := 0 },
  { event := event235866
    frameStart := 0 },
  { event := event235867
    frameStart := 0 },
  { event := event235868
    frameStart := 0 },
  { event := event235869
    frameStart := 0 },
  { event := event235870
    frameStart := 0 },
  { event := event235871
    frameStart := 0 }
]

def eventLeaf14742 : Array AnnotatedEvent := #[
  { event := event235872
    frameStart := 0 },
  { event := event235873
    frameStart := 235873 },
  { event := event235874
    frameStart := 235873 },
  { event := event235875
    frameStart := 235873 },
  { event := event235876
    frameStart := 235873 },
  { event := event235877
    frameStart := 235873 },
  { event := event235878
    frameStart := 235873 },
  { event := event235879
    frameStart := 235873 },
  { event := event235880
    frameStart := 235873 },
  { event := event235881
    frameStart := 235873 },
  { event := event235882
    frameStart := 235873 },
  { event := event235883
    frameStart := 235873 },
  { event := event235884
    frameStart := 235873 },
  { event := event235885
    frameStart := 235873 },
  { event := event235886
    frameStart := 235873 },
  { event := event235887
    frameStart := 235873 }
]

def eventLeaf14743 : Array AnnotatedEvent := #[
  { event := event235888
    frameStart := 235873 },
  { event := event235889
    frameStart := 235873 },
  { event := event235890
    frameStart := 235873 },
  { event := event235891
    frameStart := 235873 },
  { event := event235892
    frameStart := 235873 },
  { event := event235893
    frameStart := 235873 },
  { event := event235894
    frameStart := 235873 },
  { event := event235895
    frameStart := 235873 },
  { event := event235896
    frameStart := 235873 },
  { event := event235897
    frameStart := 235873 },
  { event := event235898
    frameStart := 235873 },
  { event := event235899
    frameStart := 235873 },
  { event := event235900
    frameStart := 235873 },
  { event := event235901
    frameStart := 235873 },
  { event := event235902
    frameStart := 235873 },
  { event := event235903
    frameStart := 235873 }
]

def eventLeaf14744 : Array AnnotatedEvent := #[
  { event := event235904
    frameStart := 235873 },
  { event := event235905
    frameStart := 235873 },
  { event := event235906
    frameStart := 235873 },
  { event := event235907
    frameStart := 235873 },
  { event := event235908
    frameStart := 235873 },
  { event := event235909
    frameStart := 235873 },
  { event := event235910
    frameStart := 235873 },
  { event := event235911
    frameStart := 235873 },
  { event := event235912
    frameStart := 235873 },
  { event := event235913
    frameStart := 235873 },
  { event := event235914
    frameStart := 235873 },
  { event := event235915
    frameStart := 235873 },
  { event := event235916
    frameStart := 235873 },
  { event := event235917
    frameStart := 235873 },
  { event := event235918
    frameStart := 235873 },
  { event := event235919
    frameStart := 235873 }
]

def eventLeaf14745 : Array AnnotatedEvent := #[
  { event := event235920
    frameStart := 235873 },
  { event := event235921
    frameStart := 235873 },
  { event := event235922
    frameStart := 235873 },
  { event := event235923
    frameStart := 235873 },
  { event := event235924
    frameStart := 235873 },
  { event := event235925
    frameStart := 235873 },
  { event := event235926
    frameStart := 235873 },
  { event := event235927
    frameStart := 235927 },
  { event := event235928
    frameStart := 235927 },
  { event := event235929
    frameStart := 235927 },
  { event := event235930
    frameStart := 235927 },
  { event := event235931
    frameStart := 235927 },
  { event := event235932
    frameStart := 235927 },
  { event := event235933
    frameStart := 235927 },
  { event := event235934
    frameStart := 235927 },
  { event := event235935
    frameStart := 235927 }
]

def eventLeaf14746 : Array AnnotatedEvent := #[
  { event := event235936
    frameStart := 235927 },
  { event := event235937
    frameStart := 235927 },
  { event := event235938
    frameStart := 235927 },
  { event := event235939
    frameStart := 235927 },
  { event := event235940
    frameStart := 235927 },
  { event := event235941
    frameStart := 235927 },
  { event := event235942
    frameStart := 235927 },
  { event := event235943
    frameStart := 235927 },
  { event := event235944
    frameStart := 235927 },
  { event := event235945
    frameStart := 235927 },
  { event := event235946
    frameStart := 235927 },
  { event := event235947
    frameStart := 235927 },
  { event := event235948
    frameStart := 235927 },
  { event := event235949
    frameStart := 235927 },
  { event := event235950
    frameStart := 235927 },
  { event := event235951
    frameStart := 235927 }
]

def eventLeaf14747 : Array AnnotatedEvent := #[
  { event := event235952
    frameStart := 235927 },
  { event := event235953
    frameStart := 235927 },
  { event := event235954
    frameStart := 235927 },
  { event := event235955
    frameStart := 235927 },
  { event := event235956
    frameStart := 235927 },
  { event := event235957
    frameStart := 235927 },
  { event := event235958
    frameStart := 235927 },
  { event := event235959
    frameStart := 235927 },
  { event := event235960
    frameStart := 235927 },
  { event := event235961
    frameStart := 235927 },
  { event := event235962
    frameStart := 235927 },
  { event := event235963
    frameStart := 235927 },
  { event := event235964
    frameStart := 235927 },
  { event := event235965
    frameStart := 235927 },
  { event := event235966
    frameStart := 235927 },
  { event := event235967
    frameStart := 235927 }
]

def eventLeaf14748 : Array AnnotatedEvent := #[
  { event := event235968
    frameStart := 235927 },
  { event := event235969
    frameStart := 235927 },
  { event := event235970
    frameStart := 235927 },
  { event := event235971
    frameStart := 235927 },
  { event := event235972
    frameStart := 235927 },
  { event := event235973
    frameStart := 235927 },
  { event := event235974
    frameStart := 235927 },
  { event := event235975
    frameStart := 235927 },
  { event := event235976
    frameStart := 235927 },
  { event := event235977
    frameStart := 235927 },
  { event := event235978
    frameStart := 235927 },
  { event := event235979
    frameStart := 235927 },
  { event := event235980
    frameStart := 235927 },
  { event := event235981
    frameStart := 235927 },
  { event := event235982
    frameStart := 235927 },
  { event := event235983
    frameStart := 235927 }
]

def eventLeaf14749 : Array AnnotatedEvent := #[
  { event := event235984
    frameStart := 235927 },
  { event := event235985
    frameStart := 235927 },
  { event := event235986
    frameStart := 235927 },
  { event := event235987
    frameStart := 235927 },
  { event := event235988
    frameStart := 235927 },
  { event := event235989
    frameStart := 235927 },
  { event := event235990
    frameStart := 235927 },
  { event := event235991
    frameStart := 235927 },
  { event := event235992
    frameStart := 235927 },
  { event := event235993
    frameStart := 235927 },
  { event := event235994
    frameStart := 235927 },
  { event := event235995
    frameStart := 235927 },
  { event := event235996
    frameStart := 235927 },
  { event := event235997
    frameStart := 235927 },
  { event := event235998
    frameStart := 235927 },
  { event := event235999
    frameStart := 235927 }
]

def eventLeaf14750 : Array AnnotatedEvent := #[
  { event := event236000
    frameStart := 235927 },
  { event := event236001
    frameStart := 235927 },
  { event := event236002
    frameStart := 235927 },
  { event := event236003
    frameStart := 235927 },
  { event := event236004
    frameStart := 235927 },
  { event := event236005
    frameStart := 235927 },
  { event := event236006
    frameStart := 235927 },
  { event := event236007
    frameStart := 235927 },
  { event := event236008
    frameStart := 235927 },
  { event := event236009
    frameStart := 235927 },
  { event := event236010
    frameStart := 235927 },
  { event := event236011
    frameStart := 235927 },
  { event := event236012
    frameStart := 235927 },
  { event := event236013
    frameStart := 235927 },
  { event := event236014
    frameStart := 235927 },
  { event := event236015
    frameStart := 235927 }
]

def eventLeaf14751 : Array AnnotatedEvent := #[
  { event := event236016
    frameStart := 235927 },
  { event := event236017
    frameStart := 235927 },
  { event := event236018
    frameStart := 235927 },
  { event := event236019
    frameStart := 235927 },
  { event := event236020
    frameStart := 235927 },
  { event := event236021
    frameStart := 235927 },
  { event := event236022
    frameStart := 235927 },
  { event := event236023
    frameStart := 235927 },
  { event := event236024
    frameStart := 235927 },
  { event := event236025
    frameStart := 235927 },
  { event := event236026
    frameStart := 235927 },
  { event := event236027
    frameStart := 235927 },
  { event := event236028
    frameStart := 235927 },
  { event := event236029
    frameStart := 235927 },
  { event := event236030
    frameStart := 235927 },
  { event := event236031
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events921
