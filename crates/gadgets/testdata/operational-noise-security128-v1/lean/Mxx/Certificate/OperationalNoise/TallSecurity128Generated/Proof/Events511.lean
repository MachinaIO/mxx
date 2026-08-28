import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events511

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event130816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event130817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39698⟩⟩) 0 ⟨5523⟩ 130816

def event130818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39698⟩⟩) (.authority (.programFamilyFact))

def exact130819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩]

theorem exact130819RawTermsValid :
    exact130819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39698⟩⟩) exact130819RawTerms (.finite 46) 130818 .exactZero (none)

def event130820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14121⟩⟩) 0 ⟨5523⟩ 130816

def event130821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14121⟩⟩) (.authority (.programFamilyFact))

def exact130822RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩], []⟩, (1)⟩]

theorem exact130822RawTermsValid :
    exact130822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14121⟩⟩) exact130822RawTerms (.finite 46) 130821 .exactZero (none)

def event130823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 0 ⟨14121⟩ 130822

def event130824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 1 ⟨39698⟩ 130819

def event130825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39699⟩⟩) (.product (.predecessor 0 130823 .coefficient) (.predecessor 1 130824 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event130826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39699⟩⟩, .operator (⟨130822, 0⟩, ⟨130819, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩)

def exact130827RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩]

theorem exact130827RawTermsValid :
    exact130827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130827 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39699⟩⟩) exact130827RawTerms (.finite 2116) 130825 .exactZero (none)

def event130828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39700⟩⟩) 0 ⟨39699⟩ 130827

def event130829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.identity (.predecessor 0 130828 .coefficient))

def event130830 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.finite 2116)

def event130831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40076⟩⟩) 0 ⟨39700⟩ 130830

def event130832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40076⟩⟩) (.authority (.programFamilyFact))

def exact130833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], []⟩, (1)⟩]

theorem exact130833RawTermsValid :
    exact130833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40076⟩⟩) exact130833RawTerms (.finite 46) 130832 .exactZero (none)

def event130834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40077⟩⟩) 0 ⟨40076⟩ 130833

def event130835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.identity (.predecessor 0 130834 .coefficient))

def event130836 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.finite 46)

def event130837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41223⟩⟩) 0 ⟨40077⟩ 130836

def event130838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41223⟩⟩) (.authority (.programFamilyFact))

def event130839 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41223⟩⟩) (.finite 3720)

def event130840 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event130841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41224⟩⟩) 0 ⟨7177⟩ 130840

def event130842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41224⟩⟩) 1 ⟨41223⟩ 130839

def event130843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41224⟩⟩) (.authority (.operator))

def exact130844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩]

theorem exact130844RawTermsValid :
    exact130844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41224⟩⟩) exact130844RawTerms .large 130843 .exactZero (none)

def event130845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41883⟩⟩) 0 ⟨41224⟩ 130844

def event130846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41883⟩⟩) (.authority (.operator))

def exact130847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩]

theorem exact130847RawTermsValid :
    exact130847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41883⟩⟩) exact130847RawTerms (.finite 8192) 130846 .exactZero (none)

def event130848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event130849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event130850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41450⟩⟩) 0 ⟨40077⟩ 130836

def event130851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41450⟩⟩) 1 ⟨136⟩ 130849

def event130852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41450⟩⟩) (.sum [.predecessor 0 130850 .coefficient, .predecessor 1 130851 .coefficient])

def event130853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41450⟩⟩) (.finite 46)

def event130854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41451⟩⟩) 0 ⟨41450⟩ 130853

def event130855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41451⟩⟩) (.identity (.predecessor 0 130854 .coefficient))

def exact130856RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], []⟩, (1)⟩]

theorem exact130856RawTermsValid :
    exact130856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41451⟩⟩) exact130856RawTerms (.finite 46) 130855 .exactZero (none)

def event130857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact130858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130858RawTermsValid :
    exact130858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact130858RawTerms .large 130857 .exactZero (none)

def event130859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41452⟩⟩) 0 ⟨6908⟩ 130858

def event130860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41452⟩⟩) 1 ⟨41451⟩ 130856

def event130861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41452⟩⟩) (.product (.predecessor 0 130859 .coefficient) (.predecessor 1 130860 .coefficient) (⟨false, false, none, none, none⟩))

def event130862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41452⟩⟩, .operator (⟨130858, 0⟩, ⟨130856, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact130863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130863RawTermsValid :
    exact130863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41452⟩⟩) exact130863RawTerms .large 130861 .exactZero (none)

def event130864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 130840

def event130865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact130866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact130866RawTermsValid :
    exact130866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact130866RawTerms .large 130865 .exactZero (none)

def event130867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41453⟩⟩) 0 ⟨7193⟩ 130866

def event130868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41453⟩⟩) 1 ⟨41452⟩ 130863

def event130869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41453⟩⟩) (.sum [.predecessor 0 130867 .coefficient, .predecessor 1 130868 .coefficient])

def exact130870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130870RawTermsValid :
    exact130870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41453⟩⟩) exact130870RawTerms .large 130869 .exactZero (none)

def event130871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41884⟩⟩) 0 ⟨41453⟩ 130870

def event130872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41884⟩⟩) 1 ⟨41883⟩ 130847

def event130873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41884⟩⟩) (.product (.predecessor 0 130871 .coefficient) (.predecessor 1 130872 .coefficient) (⟨false, false, none, none, none⟩))

def event130874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41884⟩⟩, .operator (⟨130870, 0⟩, ⟨130847, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩)

def event130875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41884⟩⟩, .operator (⟨130870, 1⟩, ⟨130847, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩)

def event130876 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41884⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41883⟩⟩) ⟨41224⟩ 130844)

def event130877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41884⟩⟩, .relation 130876 0, ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (-1)⟩)

def exact130878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (-1)⟩]

theorem exact130878RawTermsValid :
    exact130878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41884⟩⟩) exact130878RawTerms .large 130873 .exactZero (none)

def event130879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40270⟩⟩) 0 ⟨40077⟩ 130836

def event130880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40270⟩⟩) (.authority (.programFamilyFact))

def exact130881RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩]

theorem exact130881RawTermsValid :
    exact130881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40270⟩⟩) exact130881RawTerms (.finite 46) 130880 .exactZero (none)

def event130882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40272⟩⟩) 0 ⟨6908⟩ 130858

def event130883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40272⟩⟩) 1 ⟨40270⟩ 130881

def event130884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40272⟩⟩) (.product (.predecessor 0 130882 .coefficient) (.predecessor 1 130883 .coefficient) (⟨false, true, none, none, some 1⟩))

def event130885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40272⟩⟩, .operator (⟨130858, 0⟩, ⟨130881, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact130886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact130886RawTermsValid :
    exact130886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40272⟩⟩) exact130886RawTerms .large 130884 .exactZero (none)

def event130887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7225⟩⟩) 0 ⟨7177⟩ 130840

def event130888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7225⟩⟩) (.authority (.operator))

def exact130889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩]

theorem exact130889RawTermsValid :
    exact130889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130889 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7225⟩⟩) exact130889RawTerms .large 130888 .exactZero (none)

def event130890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40273⟩⟩) 0 ⟨7225⟩ 130889

def event130891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40273⟩⟩) 1 ⟨40272⟩ 130886

def event130892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40273⟩⟩) (.sum [.predecessor 0 130890 .coefficient, .predecessor 1 130891 .coefficient])

def exact130893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130893RawTermsValid :
    exact130893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40273⟩⟩) exact130893RawTerms .large 130892 .exactZero (none)

def event130894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41888⟩⟩) 0 ⟨40273⟩ 130893

def event130895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41888⟩⟩) 1 ⟨41884⟩ 130878

def event130896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41888⟩⟩) (.sum [.predecessor 0 130894 .coefficient, .predecessor 1 130895 .coefficient])

def exact130897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130897RawTermsValid :
    exact130897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41888⟩⟩) exact130897RawTerms .large 130896 .exactZero (none)

def event130898 : Event := .preFoldPolynomial 130897 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact130899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event130899 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41888⟩⟩) 130898 exact130899RawTerms .large 130896 .exactZero (none)

def event130900 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40077⟩⟩) ⟨⟨104⟩, ⟨86⟩, ⟨135⟩⟩ ⟨130742, 130900⟩

def event130901 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40775⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩) (1) 0 2 (.universal 130900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40772⟩⟩]⟩) (none) 130899)

def event130902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40775⟩⟩, .relation 130901 1, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩)

def event130903 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40775⟩⟩, .relation 130901 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩)

def event130904 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40775⟩⟩, .relation 130901 2, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩)

def event130905 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40775⟩⟩, .relation 130901 3, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact130906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130906RawTermsValid :
    exact130906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40775⟩⟩) exact130906RawTerms .large 130738 (.finite 202072841853861888) (some (130740))

def event130907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41886⟩⟩) 0 ⟨40775⟩ 130906

def event130908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41886⟩⟩) 1 ⟨41885⟩ 130728

def event130909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41886⟩⟩) (.sum [.predecessor 0 130907 .coefficient, .predecessor 1 130908 .coefficient])

def event130910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41886⟩⟩, .operator (⟨130906, 0⟩, ⟨130728, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41883⟩⟩]⟩, (1)⟩)

def event130911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41886⟩⟩, .operator (⟨130906, 2⟩, ⟨130728, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41224⟩⟩]⟩, (-1)⟩)

def event130912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41886⟩⟩) (.sum [.result 130906 .summary, .result 130728 .summary])

def exact130913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130913RawTermsValid :
    exact130913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41886⟩⟩) exact130913RawTerms .large 130909 (.finite 32193129122288829188810200055808) (some (130912))

def event130914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41887⟩⟩) 0 ⟨41886⟩ 130913

def event130915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41887⟩⟩) 1 ⟨7160⟩ 15602

def event130916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41887⟩⟩) (.product (.predecessor 0 130914 .coefficient) (.predecessor 1 130915 .coefficient) (⟨false, false, none, none, none⟩))

def event130917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41887⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) [⟨.result 15598 .coefficient, false, none⟩])

def event130918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41887⟩⟩) (.product (.result 130913 .summary) (.transfer 130917) (⟨false, false, none, none, none⟩))

def event130919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41887⟩⟩, .operator (⟨130913, 0⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event130920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41887⟩⟩, .operator (⟨130913, 1⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event130921 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41887⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595)

def event130922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41887⟩⟩, .relation 130921 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact130923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact130923RawTermsValid :
    exact130923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41887⟩⟩) exact130923RawTerms .large 130916 (.finite 345671091840339265080175045977281837137920) (some (130918))

def event130924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38544⟩⟩) 0 ⟨7177⟩ 15500

def event130925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38544⟩⟩) 1 ⟨38543⟩ 121700

def event130926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38544⟩⟩) (.authority (.operator))

def exact130927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38544⟩⟩]⟩, (1)⟩]

theorem exact130927RawTermsValid :
    exact130927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38544⟩⟩) exact130927RawTerms .large 130926 .exactZero (none)

def event130928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39203⟩⟩) 0 ⟨38544⟩ 130927

def event130929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39203⟩⟩) (.authority (.operator))

def exact130930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩, (1)⟩]

theorem exact130930RawTermsValid :
    exact130930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39203⟩⟩) exact130930RawTerms (.finite 8192) 130929 .exactZero (none)

def event130931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39205⟩⟩) 0 ⟨38897⟩ 121984

def event130932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39205⟩⟩) 1 ⟨39203⟩ 130930

def event130933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39205⟩⟩) (.product (.predecessor 0 130931 .coefficient) (.predecessor 1 130932 .coefficient) (⟨false, false, none, none, none⟩))

def event130934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39205⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩) [⟨.result 130930 .coefficient, false, none⟩])

def event130935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39205⟩⟩) (.product (.result 121984 .summary) (.transfer 130934) (⟨false, false, none, none, none⟩))

def event130936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39205⟩⟩, .operator (⟨121984, 0⟩, ⟨130930, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩, (1)⟩)

def event130937 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39205⟩⟩, .operator (⟨121984, 1⟩, ⟨130930, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩, (-1)⟩)

def event130938 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39205⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39203⟩⟩) ⟨38544⟩ 130927)

def event130939 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39205⟩⟩, .relation 130938 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨38544⟩⟩]⟩, (-1)⟩)

def exact130940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨38544⟩⟩]⟩, (-1)⟩]

theorem exact130940RawTermsValid :
    exact130940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39205⟩⟩) exact130940RawTerms .large 130933 (.finite 32192736221397252361486566686720) (some (130935))

def event130941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38092⟩⟩) 0 ⟨37397⟩ 5439

def event130942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38092⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact130943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩]

theorem exact130943RawTermsValid :
    exact130943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38092⟩⟩) exact130943RawTerms (.finite 5647228698) 130942 .exactZero (none)

def event130944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38094⟩⟩) 0 ⟨38092⟩ 130943

def event130945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38094⟩⟩) 1 ⟨2370⟩ 4

def event130946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38094⟩⟩) (.scale (.predecessor 0 130944 .coefficient) (.value (.predecessor 1 130945 .coefficient)))

def exact130947RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩]

theorem exact130947RawTermsValid :
    exact130947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130947 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38094⟩⟩) exact130947RawTerms (.finite 5647228698) 130946 .exactZero (none)

def event130948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38095⟩⟩) 0 ⟨5527⟩ 119870

def event130949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38095⟩⟩) 1 ⟨38094⟩ 130947

def event130950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38095⟩⟩) (.product (.predecessor 0 130948 .coefficient) (.predecessor 1 130949 .coefficient) (⟨false, false, none, none, none⟩))

def event130951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38095⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩) [⟨.result 130943 .coefficient, false, none⟩])

def event130952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38095⟩⟩) (.product (.result 119870 .summary) (.transfer 130951) (⟨false, false, none, none, none⟩))

def event130953 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38095⟩⟩, .operator (⟨119870, 0⟩, ⟨130947, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩)

def event130954 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38093⟩⟩)

def event130955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event130956 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event130957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event130958 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event130959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event130960 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event130961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event130962 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event130963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 130962

def event130964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 130960

def event130965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 130963 .coefficient) (.value (.predecessor 1 130964 .coefficient)))

def event130966 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event130967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 130966

def event130968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 130958

def event130969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 130967 .coefficient, .predecessor 1 130968 .coefficient])

def event130970 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event130971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 130970

def event130972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 130956

def event130973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 130972 .coefficient))

def event130974 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event130975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37018⟩⟩) 0 ⟨5523⟩ 130974

def event130976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37018⟩⟩) (.authority (.programFamilyFact))

def exact130977RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩, (1)⟩]

theorem exact130977RawTermsValid :
    exact130977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37018⟩⟩) exact130977RawTerms (.finite 42) 130976 .exactZero (none)

def event130978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13821⟩⟩) 0 ⟨5523⟩ 130974

def event130979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13821⟩⟩) (.authority (.programFamilyFact))

def exact130980RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩], []⟩, (1)⟩]

theorem exact130980RawTermsValid :
    exact130980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13821⟩⟩) exact130980RawTerms (.finite 42) 130979 .exactZero (none)

def event130981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 0 ⟨13821⟩ 130980

def event130982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 1 ⟨37018⟩ 130977

def event130983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37019⟩⟩) (.product (.predecessor 0 130981 .coefficient) (.predecessor 1 130982 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event130984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37019⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩) [⟨.result 130980 .coefficient, true, some 1⟩, ⟨.result 130977 .coefficient, true, some 1⟩])

def event130985 : Event := .survivorFold (1) 130984

def exact130986RawTerms : List Term := []

theorem exact130986RawTermsValid :
    exact130986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37019⟩⟩) exact130986RawTerms (.finite 1764) 130983 (.finite 1764) (some (130984))

def event130987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37020⟩⟩) 0 ⟨37019⟩ 130986

def event130988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.identity (.predecessor 0 130987 .coefficient))

def event130989 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.finite 1764)

def event130990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37396⟩⟩) 0 ⟨37020⟩ 130989

def event130991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37396⟩⟩) (.authority (.programFamilyFact))

def exact130992RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37396⟩⟩], []⟩, (1)⟩]

theorem exact130992RawTermsValid :
    exact130992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37396⟩⟩) exact130992RawTerms (.finite 42) 130991 .exactZero (none)

def event130993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37397⟩⟩) 0 ⟨37396⟩ 130992

def event130994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37397⟩⟩) (.identity (.predecessor 0 130993 .coefficient))

def event130995 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37397⟩⟩) (.finite 42)

def event130996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38092⟩⟩) 0 ⟨37397⟩ 130995

def event130997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38092⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact130998RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩]

theorem exact130998RawTermsValid :
    exact130998RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event130998 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38092⟩⟩) exact130998RawTerms (.finite 5647228698) 130997 .exactZero (none)

def event130999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact131000RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact131000RawTermsValid :
    exact131000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact131000RawTerms .large 130999 .exactZero (none)

def event131001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38093⟩⟩) 0 ⟨35⟩ 131000

def event131002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38093⟩⟩) 1 ⟨38092⟩ 130998

def event131003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38093⟩⟩) (.product (.predecessor 0 131001 .coefficient) (.predecessor 1 131002 .coefficient) (⟨false, false, none, none, none⟩))

def event131004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38093⟩⟩, .operator (⟨131000, 0⟩, ⟨130998, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩)

def exact131005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩]

theorem exact131005RawTermsValid :
    exact131005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131005 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38093⟩⟩) exact131005RawTerms .large 131003 .exactZero (none)

def event131006 : Event := .preFoldPolynomial 131005 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩] .exactZero none

def exact131007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38092⟩⟩]⟩, (1)⟩]

def event131007 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨38093⟩⟩) 131006 exact131007RawTerms .large 131003 .exactZero (none)

def event131008 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39208⟩⟩)

def event131009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event131010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event131011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event131012 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event131013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event131014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event131015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event131016 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event131017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 131016

def event131018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 131014

def event131019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 131017 .coefficient) (.value (.predecessor 1 131018 .coefficient)))

def event131020 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event131021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 131020

def event131022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 131012

def event131023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 131021 .coefficient, .predecessor 1 131022 .coefficient])

def event131024 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event131025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 131024

def event131026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 131010

def event131027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 131026 .coefficient))

def event131028 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event131029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37018⟩⟩) 0 ⟨5523⟩ 131028

def event131030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37018⟩⟩) (.authority (.programFamilyFact))

def exact131031RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩, (1)⟩]

theorem exact131031RawTermsValid :
    exact131031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131031 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37018⟩⟩) exact131031RawTerms (.finite 42) 131030 .exactZero (none)

def event131032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13821⟩⟩) 0 ⟨5523⟩ 131028

def event131033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13821⟩⟩) (.authority (.programFamilyFact))

def exact131034RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩], []⟩, (1)⟩]

theorem exact131034RawTermsValid :
    exact131034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13821⟩⟩) exact131034RawTerms (.finite 42) 131033 .exactZero (none)

def event131035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 0 ⟨13821⟩ 131034

def event131036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 1 ⟨37018⟩ 131031

def event131037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37019⟩⟩) (.product (.predecessor 0 131035 .coefficient) (.predecessor 1 131036 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event131038 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37019⟩⟩, .operator (⟨131034, 0⟩, ⟨131031, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩, (1)⟩)

def exact131039RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩, (1)⟩]

theorem exact131039RawTermsValid :
    exact131039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131039 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37019⟩⟩) exact131039RawTerms (.finite 1764) 131037 .exactZero (none)

def event131040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37020⟩⟩) 0 ⟨37019⟩ 131039

def event131041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.identity (.predecessor 0 131040 .coefficient))

def event131042 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.finite 1764)

def event131043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37396⟩⟩) 0 ⟨37020⟩ 131042

def event131044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37396⟩⟩) (.authority (.programFamilyFact))

def exact131045RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37396⟩⟩], []⟩, (1)⟩]

theorem exact131045RawTermsValid :
    exact131045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131045 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37396⟩⟩) exact131045RawTerms (.finite 42) 131044 .exactZero (none)

def event131046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37397⟩⟩) 0 ⟨37396⟩ 131045

def event131047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37397⟩⟩) (.identity (.predecessor 0 131046 .coefficient))

def event131048 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37397⟩⟩) (.finite 42)

def event131049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38543⟩⟩) 0 ⟨37397⟩ 131048

def event131050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38543⟩⟩) (.authority (.programFamilyFact))

def event131051 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38543⟩⟩) (.finite 3720)

def event131052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event131053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38544⟩⟩) 0 ⟨7177⟩ 131052

def event131054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38544⟩⟩) 1 ⟨38543⟩ 131051

def event131055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38544⟩⟩) (.authority (.operator))

def exact131056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38544⟩⟩]⟩, (1)⟩]

theorem exact131056RawTermsValid :
    exact131056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38544⟩⟩) exact131056RawTerms .large 131055 .exactZero (none)

def event131057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39203⟩⟩) 0 ⟨38544⟩ 131056

def event131058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39203⟩⟩) (.authority (.operator))

def exact131059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩, (1)⟩]

theorem exact131059RawTermsValid :
    exact131059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131059 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39203⟩⟩) exact131059RawTerms (.finite 8192) 131058 .exactZero (none)

def event131060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event131061 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event131062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38770⟩⟩) 0 ⟨37397⟩ 131048

def event131063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38770⟩⟩) 1 ⟨136⟩ 131061

def event131064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38770⟩⟩) (.sum [.predecessor 0 131062 .coefficient, .predecessor 1 131063 .coefficient])

def event131065 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38770⟩⟩) (.finite 42)

def event131066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38771⟩⟩) 0 ⟨38770⟩ 131065

def event131067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38771⟩⟩) (.identity (.predecessor 0 131066 .coefficient))

def exact131068RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37396⟩⟩], []⟩, (1)⟩]

theorem exact131068RawTermsValid :
    exact131068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38771⟩⟩) exact131068RawTerms (.finite 42) 131067 .exactZero (none)

def event131069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact131070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact131070RawTermsValid :
    exact131070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event131070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact131070RawTerms .large 131069 .exactZero (none)

def event131071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38772⟩⟩) 0 ⟨6908⟩ 131070

def eventLeaf8176 : Array AnnotatedEvent := #[
  { event := event130816
    frameStart := 130796 },
  { event := event130817
    frameStart := 130796 },
  { event := event130818
    frameStart := 130796 },
  { event := event130819
    frameStart := 130796 },
  { event := event130820
    frameStart := 130796 },
  { event := event130821
    frameStart := 130796 },
  { event := event130822
    frameStart := 130796 },
  { event := event130823
    frameStart := 130796 },
  { event := event130824
    frameStart := 130796 },
  { event := event130825
    frameStart := 130796 },
  { event := event130826
    frameStart := 130796 },
  { event := event130827
    frameStart := 130796 },
  { event := event130828
    frameStart := 130796 },
  { event := event130829
    frameStart := 130796 },
  { event := event130830
    frameStart := 130796 },
  { event := event130831
    frameStart := 130796 }
]

def eventLeaf8177 : Array AnnotatedEvent := #[
  { event := event130832
    frameStart := 130796 },
  { event := event130833
    frameStart := 130796 },
  { event := event130834
    frameStart := 130796 },
  { event := event130835
    frameStart := 130796 },
  { event := event130836
    frameStart := 130796 },
  { event := event130837
    frameStart := 130796 },
  { event := event130838
    frameStart := 130796 },
  { event := event130839
    frameStart := 130796 },
  { event := event130840
    frameStart := 130796 },
  { event := event130841
    frameStart := 130796 },
  { event := event130842
    frameStart := 130796 },
  { event := event130843
    frameStart := 130796 },
  { event := event130844
    frameStart := 130796 },
  { event := event130845
    frameStart := 130796 },
  { event := event130846
    frameStart := 130796 },
  { event := event130847
    frameStart := 130796 }
]

def eventLeaf8178 : Array AnnotatedEvent := #[
  { event := event130848
    frameStart := 130796 },
  { event := event130849
    frameStart := 130796 },
  { event := event130850
    frameStart := 130796 },
  { event := event130851
    frameStart := 130796 },
  { event := event130852
    frameStart := 130796 },
  { event := event130853
    frameStart := 130796 },
  { event := event130854
    frameStart := 130796 },
  { event := event130855
    frameStart := 130796 },
  { event := event130856
    frameStart := 130796 },
  { event := event130857
    frameStart := 130796 },
  { event := event130858
    frameStart := 130796 },
  { event := event130859
    frameStart := 130796 },
  { event := event130860
    frameStart := 130796 },
  { event := event130861
    frameStart := 130796 },
  { event := event130862
    frameStart := 130796 },
  { event := event130863
    frameStart := 130796 }
]

def eventLeaf8179 : Array AnnotatedEvent := #[
  { event := event130864
    frameStart := 130796 },
  { event := event130865
    frameStart := 130796 },
  { event := event130866
    frameStart := 130796 },
  { event := event130867
    frameStart := 130796 },
  { event := event130868
    frameStart := 130796 },
  { event := event130869
    frameStart := 130796 },
  { event := event130870
    frameStart := 130796 },
  { event := event130871
    frameStart := 130796 },
  { event := event130872
    frameStart := 130796 },
  { event := event130873
    frameStart := 130796 },
  { event := event130874
    frameStart := 130796 },
  { event := event130875
    frameStart := 130796 },
  { event := event130876
    frameStart := 130796 },
  { event := event130877
    frameStart := 130796 },
  { event := event130878
    frameStart := 130796 },
  { event := event130879
    frameStart := 130796 }
]

def eventLeaf8180 : Array AnnotatedEvent := #[
  { event := event130880
    frameStart := 130796 },
  { event := event130881
    frameStart := 130796 },
  { event := event130882
    frameStart := 130796 },
  { event := event130883
    frameStart := 130796 },
  { event := event130884
    frameStart := 130796 },
  { event := event130885
    frameStart := 130796 },
  { event := event130886
    frameStart := 130796 },
  { event := event130887
    frameStart := 130796 },
  { event := event130888
    frameStart := 130796 },
  { event := event130889
    frameStart := 130796 },
  { event := event130890
    frameStart := 130796 },
  { event := event130891
    frameStart := 130796 },
  { event := event130892
    frameStart := 130796 },
  { event := event130893
    frameStart := 130796 },
  { event := event130894
    frameStart := 130796 },
  { event := event130895
    frameStart := 130796 }
]

def eventLeaf8181 : Array AnnotatedEvent := #[
  { event := event130896
    frameStart := 130796 },
  { event := event130897
    frameStart := 130796 },
  { event := event130898
    frameStart := 130796 },
  { event := event130899
    frameStart := 130796 },
  { event := event130900
    frameStart := 0 },
  { event := event130901
    frameStart := 0 },
  { event := event130902
    frameStart := 0 },
  { event := event130903
    frameStart := 0 },
  { event := event130904
    frameStart := 0 },
  { event := event130905
    frameStart := 0 },
  { event := event130906
    frameStart := 0 },
  { event := event130907
    frameStart := 0 },
  { event := event130908
    frameStart := 0 },
  { event := event130909
    frameStart := 0 },
  { event := event130910
    frameStart := 0 },
  { event := event130911
    frameStart := 0 }
]

def eventLeaf8182 : Array AnnotatedEvent := #[
  { event := event130912
    frameStart := 0 },
  { event := event130913
    frameStart := 0 },
  { event := event130914
    frameStart := 0 },
  { event := event130915
    frameStart := 0 },
  { event := event130916
    frameStart := 0 },
  { event := event130917
    frameStart := 0 },
  { event := event130918
    frameStart := 0 },
  { event := event130919
    frameStart := 0 },
  { event := event130920
    frameStart := 0 },
  { event := event130921
    frameStart := 0 },
  { event := event130922
    frameStart := 0 },
  { event := event130923
    frameStart := 0 },
  { event := event130924
    frameStart := 0 },
  { event := event130925
    frameStart := 0 },
  { event := event130926
    frameStart := 0 },
  { event := event130927
    frameStart := 0 }
]

def eventLeaf8183 : Array AnnotatedEvent := #[
  { event := event130928
    frameStart := 0 },
  { event := event130929
    frameStart := 0 },
  { event := event130930
    frameStart := 0 },
  { event := event130931
    frameStart := 0 },
  { event := event130932
    frameStart := 0 },
  { event := event130933
    frameStart := 0 },
  { event := event130934
    frameStart := 0 },
  { event := event130935
    frameStart := 0 },
  { event := event130936
    frameStart := 0 },
  { event := event130937
    frameStart := 0 },
  { event := event130938
    frameStart := 0 },
  { event := event130939
    frameStart := 0 },
  { event := event130940
    frameStart := 0 },
  { event := event130941
    frameStart := 0 },
  { event := event130942
    frameStart := 0 },
  { event := event130943
    frameStart := 0 }
]

def eventLeaf8184 : Array AnnotatedEvent := #[
  { event := event130944
    frameStart := 0 },
  { event := event130945
    frameStart := 0 },
  { event := event130946
    frameStart := 0 },
  { event := event130947
    frameStart := 0 },
  { event := event130948
    frameStart := 0 },
  { event := event130949
    frameStart := 0 },
  { event := event130950
    frameStart := 0 },
  { event := event130951
    frameStart := 0 },
  { event := event130952
    frameStart := 0 },
  { event := event130953
    frameStart := 0 },
  { event := event130954
    frameStart := 130954 },
  { event := event130955
    frameStart := 130954 },
  { event := event130956
    frameStart := 130954 },
  { event := event130957
    frameStart := 130954 },
  { event := event130958
    frameStart := 130954 },
  { event := event130959
    frameStart := 130954 }
]

def eventLeaf8185 : Array AnnotatedEvent := #[
  { event := event130960
    frameStart := 130954 },
  { event := event130961
    frameStart := 130954 },
  { event := event130962
    frameStart := 130954 },
  { event := event130963
    frameStart := 130954 },
  { event := event130964
    frameStart := 130954 },
  { event := event130965
    frameStart := 130954 },
  { event := event130966
    frameStart := 130954 },
  { event := event130967
    frameStart := 130954 },
  { event := event130968
    frameStart := 130954 },
  { event := event130969
    frameStart := 130954 },
  { event := event130970
    frameStart := 130954 },
  { event := event130971
    frameStart := 130954 },
  { event := event130972
    frameStart := 130954 },
  { event := event130973
    frameStart := 130954 },
  { event := event130974
    frameStart := 130954 },
  { event := event130975
    frameStart := 130954 }
]

def eventLeaf8186 : Array AnnotatedEvent := #[
  { event := event130976
    frameStart := 130954 },
  { event := event130977
    frameStart := 130954 },
  { event := event130978
    frameStart := 130954 },
  { event := event130979
    frameStart := 130954 },
  { event := event130980
    frameStart := 130954 },
  { event := event130981
    frameStart := 130954 },
  { event := event130982
    frameStart := 130954 },
  { event := event130983
    frameStart := 130954 },
  { event := event130984
    frameStart := 130954 },
  { event := event130985
    frameStart := 130954 },
  { event := event130986
    frameStart := 130954 },
  { event := event130987
    frameStart := 130954 },
  { event := event130988
    frameStart := 130954 },
  { event := event130989
    frameStart := 130954 },
  { event := event130990
    frameStart := 130954 },
  { event := event130991
    frameStart := 130954 }
]

def eventLeaf8187 : Array AnnotatedEvent := #[
  { event := event130992
    frameStart := 130954 },
  { event := event130993
    frameStart := 130954 },
  { event := event130994
    frameStart := 130954 },
  { event := event130995
    frameStart := 130954 },
  { event := event130996
    frameStart := 130954 },
  { event := event130997
    frameStart := 130954 },
  { event := event130998
    frameStart := 130954 },
  { event := event130999
    frameStart := 130954 },
  { event := event131000
    frameStart := 130954 },
  { event := event131001
    frameStart := 130954 },
  { event := event131002
    frameStart := 130954 },
  { event := event131003
    frameStart := 130954 },
  { event := event131004
    frameStart := 130954 },
  { event := event131005
    frameStart := 130954 },
  { event := event131006
    frameStart := 130954 },
  { event := event131007
    frameStart := 130954 }
]

def eventLeaf8188 : Array AnnotatedEvent := #[
  { event := event131008
    frameStart := 131008 },
  { event := event131009
    frameStart := 131008 },
  { event := event131010
    frameStart := 131008 },
  { event := event131011
    frameStart := 131008 },
  { event := event131012
    frameStart := 131008 },
  { event := event131013
    frameStart := 131008 },
  { event := event131014
    frameStart := 131008 },
  { event := event131015
    frameStart := 131008 },
  { event := event131016
    frameStart := 131008 },
  { event := event131017
    frameStart := 131008 },
  { event := event131018
    frameStart := 131008 },
  { event := event131019
    frameStart := 131008 },
  { event := event131020
    frameStart := 131008 },
  { event := event131021
    frameStart := 131008 },
  { event := event131022
    frameStart := 131008 },
  { event := event131023
    frameStart := 131008 }
]

def eventLeaf8189 : Array AnnotatedEvent := #[
  { event := event131024
    frameStart := 131008 },
  { event := event131025
    frameStart := 131008 },
  { event := event131026
    frameStart := 131008 },
  { event := event131027
    frameStart := 131008 },
  { event := event131028
    frameStart := 131008 },
  { event := event131029
    frameStart := 131008 },
  { event := event131030
    frameStart := 131008 },
  { event := event131031
    frameStart := 131008 },
  { event := event131032
    frameStart := 131008 },
  { event := event131033
    frameStart := 131008 },
  { event := event131034
    frameStart := 131008 },
  { event := event131035
    frameStart := 131008 },
  { event := event131036
    frameStart := 131008 },
  { event := event131037
    frameStart := 131008 },
  { event := event131038
    frameStart := 131008 },
  { event := event131039
    frameStart := 131008 }
]

def eventLeaf8190 : Array AnnotatedEvent := #[
  { event := event131040
    frameStart := 131008 },
  { event := event131041
    frameStart := 131008 },
  { event := event131042
    frameStart := 131008 },
  { event := event131043
    frameStart := 131008 },
  { event := event131044
    frameStart := 131008 },
  { event := event131045
    frameStart := 131008 },
  { event := event131046
    frameStart := 131008 },
  { event := event131047
    frameStart := 131008 },
  { event := event131048
    frameStart := 131008 },
  { event := event131049
    frameStart := 131008 },
  { event := event131050
    frameStart := 131008 },
  { event := event131051
    frameStart := 131008 },
  { event := event131052
    frameStart := 131008 },
  { event := event131053
    frameStart := 131008 },
  { event := event131054
    frameStart := 131008 },
  { event := event131055
    frameStart := 131008 }
]

def eventLeaf8191 : Array AnnotatedEvent := #[
  { event := event131056
    frameStart := 131008 },
  { event := event131057
    frameStart := 131008 },
  { event := event131058
    frameStart := 131008 },
  { event := event131059
    frameStart := 131008 },
  { event := event131060
    frameStart := 131008 },
  { event := event131061
    frameStart := 131008 },
  { event := event131062
    frameStart := 131008 },
  { event := event131063
    frameStart := 131008 },
  { event := event131064
    frameStart := 131008 },
  { event := event131065
    frameStart := 131008 },
  { event := event131066
    frameStart := 131008 },
  { event := event131067
    frameStart := 131008 },
  { event := event131068
    frameStart := 131008 },
  { event := event131069
    frameStart := 131008 },
  { event := event131070
    frameStart := 131008 },
  { event := event131071
    frameStart := 131008 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events511
