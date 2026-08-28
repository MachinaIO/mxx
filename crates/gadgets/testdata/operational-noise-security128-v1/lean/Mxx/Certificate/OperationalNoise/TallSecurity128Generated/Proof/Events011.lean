import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events011

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event2816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67100⟩⟩) (.sum [.predecessor 0 2814 .coefficient, .predecessor 1 2815 .coefficient])

def exact2817RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩, (1)⟩]

theorem exact2817RawTermsValid :
    exact2817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67100⟩⟩) exact2817RawTerms (.finite 1059) 2816 .exactZero (none)

def event2818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67101⟩⟩) 0 ⟨67100⟩ 2817

def event2819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.identity (.predecessor 0 2818 .coefficient))

def event2820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.finite 1059)

def event2821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67606⟩⟩) 0 ⟨67101⟩ 2820

def event2822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67606⟩⟩) (.authority (.programFamilyFact))

def exact2823RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (1)⟩]

theorem exact2823RawTermsValid :
    exact2823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67606⟩⟩) exact2823RawTerms (.finite 18) 2822 .exactZero (none)

def event2824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67607⟩⟩) 0 ⟨67606⟩ 2823

def event2825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67607⟩⟩) 1 ⟨6774⟩ 36

def event2826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67607⟩⟩) (.product (.predecessor 0 2824 .coefficient) (.predecessor 1 2825 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67607⟩⟩, .operator (⟨2823, 0⟩, ⟨36, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (1)⟩)

def exact2828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (1)⟩]

theorem exact2828RawTermsValid :
    exact2828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67607⟩⟩) exact2828RawTerms (.finite 4222381728938650955397720) 2826 .exactZero (none)

def event2829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48450⟩⟩) 0 ⟨48205⟩ 2355

def event2830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48450⟩⟩) (.authority (.programFamilyFact))

def exact2831RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩]

theorem exact2831RawTermsValid :
    exact2831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48450⟩⟩) exact2831RawTerms (.finite 60) 2830 .exactZero (none)

def event2832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48451⟩⟩) 0 ⟨48450⟩ 2831

def event2833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48451⟩⟩) 1 ⟨6800⟩ 543

def event2834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48451⟩⟩) (.product (.predecessor 0 2832 .coefficient) (.predecessor 1 2833 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48451⟩⟩, .operator (⟨2831, 0⟩, ⟨543, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩)

def exact2836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩]

theorem exact2836RawTermsValid :
    exact2836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48451⟩⟩) exact2836RawTerms (.finite 230731242018505516688400) 2834 .exactZero (none)

def event2837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45770⟩⟩) 0 ⟨45525⟩ 2378

def event2838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45770⟩⟩) (.authority (.programFamilyFact))

def exact2839RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩]

theorem exact2839RawTermsValid :
    exact2839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45770⟩⟩) exact2839RawTerms (.finite 58) 2838 .exactZero (none)

def event2840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45771⟩⟩) 0 ⟨45770⟩ 2839

def event2841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45771⟩⟩) 1 ⟨6807⟩ 553

def event2842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45771⟩⟩) (.product (.predecessor 0 2840 .coefficient) (.predecessor 1 2841 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45771⟩⟩, .operator (⟨2839, 0⟩, ⟨553, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩)

def exact2844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩]

theorem exact2844RawTermsValid :
    exact2844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45771⟩⟩) exact2844RawTerms (.finite 230600885384596756509480) 2842 .exactZero (none)

def event2845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43093⟩⟩) 0 ⟨42845⟩ 2401

def event2846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43093⟩⟩) (.authority (.programFamilyFact))

def exact2847RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩]

theorem exact2847RawTermsValid :
    exact2847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43093⟩⟩) exact2847RawTerms (.finite 52) 2846 .exactZero (none)

def event2848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43094⟩⟩) 0 ⟨43093⟩ 2847

def event2849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43094⟩⟩) 1 ⟨6817⟩ 563

def event2850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43094⟩⟩) (.product (.predecessor 0 2848 .coefficient) (.predecessor 1 2849 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43094⟩⟩, .operator (⟨2847, 0⟩, ⟨563, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩)

def exact2852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩]

theorem exact2852RawTermsValid :
    exact2852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43094⟩⟩) exact2852RawTerms (.finite 230150786063741980797360) 2850 .exactZero (none)

def event2853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40413⟩⟩) 0 ⟨40165⟩ 2424

def event2854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40413⟩⟩) (.authority (.programFamilyFact))

def exact2855RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩]

theorem exact2855RawTermsValid :
    exact2855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40413⟩⟩) exact2855RawTerms (.finite 46) 2854 .exactZero (none)

def event2856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40414⟩⟩) 0 ⟨40413⟩ 2855

def event2857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40414⟩⟩) 1 ⟨6828⟩ 573

def event2858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40414⟩⟩) (.product (.predecessor 0 2856 .coefficient) (.predecessor 1 2857 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40414⟩⟩, .operator (⟨2855, 0⟩, ⟨573, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩)

def exact2860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩]

theorem exact2860RawTermsValid :
    exact2860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40414⟩⟩) exact2860RawTerms (.finite 229585767767349815541720) 2858 .exactZero (none)

def event2861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37730⟩⟩) 0 ⟨37485⟩ 2447

def event2862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37730⟩⟩) (.authority (.programFamilyFact))

def exact2863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩]

theorem exact2863RawTermsValid :
    exact2863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37730⟩⟩) exact2863RawTerms (.finite 42) 2862 .exactZero (none)

def event2864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37731⟩⟩) 0 ⟨37730⟩ 2863

def event2865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37731⟩⟩) 1 ⟨6838⟩ 583

def event2866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37731⟩⟩) (.product (.predecessor 0 2864 .coefficient) (.predecessor 1 2865 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37731⟩⟩, .operator (⟨2863, 0⟩, ⟨583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩)

def exact2868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩]

theorem exact2868RawTermsValid :
    exact2868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37731⟩⟩) exact2868RawTerms (.finite 229121489167213617734760) 2866 .exactZero (none)

def event2869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35050⟩⟩) 0 ⟨34805⟩ 2470

def event2870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35050⟩⟩) (.authority (.programFamilyFact))

def exact2871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩]

theorem exact2871RawTermsValid :
    exact2871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35050⟩⟩) exact2871RawTerms (.finite 40) 2870 .exactZero (none)

def event2872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35051⟩⟩) 0 ⟨35050⟩ 2871

def event2873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35051⟩⟩) 1 ⟨6842⟩ 593

def event2874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35051⟩⟩) (.product (.predecessor 0 2872 .coefficient) (.predecessor 1 2873 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35051⟩⟩, .operator (⟨2871, 0⟩, ⟨593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩)

def exact2876RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩]

theorem exact2876RawTermsValid :
    exact2876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2876 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35051⟩⟩) exact2876RawTerms (.finite 228855378262257504357600) 2874 .exactZero (none)

def event2877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29393⟩⟩) 0 ⟨29145⟩ 2493

def event2878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29393⟩⟩) (.authority (.programFamilyFact))

def exact2879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩]

theorem exact2879RawTermsValid :
    exact2879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29393⟩⟩) exact2879RawTerms (.finite 36) 2878 .exactZero (none)

def event2880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29394⟩⟩) 0 ⟨29393⟩ 2879

def event2881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29394⟩⟩) 1 ⟨6857⟩ 603

def event2882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29394⟩⟩) (.product (.predecessor 0 2880 .coefficient) (.predecessor 1 2881 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29394⟩⟩, .operator (⟨2879, 0⟩, ⟨603, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩)

def exact2884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩]

theorem exact2884RawTermsValid :
    exact2884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29394⟩⟩) exact2884RawTerms (.finite 228236850212900051643120) 2882 .exactZero (none)

def event2885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26713⟩⟩) 0 ⟨26465⟩ 2516

def event2886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26713⟩⟩) (.authority (.programFamilyFact))

def exact2887RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩]

theorem exact2887RawTermsValid :
    exact2887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2887 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26713⟩⟩) exact2887RawTerms (.finite 30) 2886 .exactZero (none)

def event2888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26714⟩⟩) 0 ⟨26713⟩ 2887

def event2889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26714⟩⟩) 1 ⟨6860⟩ 613

def event2890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26714⟩⟩) (.product (.predecessor 0 2888 .coefficient) (.predecessor 1 2889 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26714⟩⟩, .operator (⟨2887, 0⟩, ⟨613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩)

def exact2892RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩]

theorem exact2892RawTermsValid :
    exact2892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26714⟩⟩) exact2892RawTerms (.finite 227009770373045750290200) 2890 .exactZero (none)

def event2893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67078⟩⟩) 0 ⟨65845⟩ 2539

def event2894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67078⟩⟩) (.authority (.programFamilyFact))

def exact2895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact2895RawTermsValid :
    exact2895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67078⟩⟩) exact2895RawTerms (.finite 28) 2894 .exactZero (none)

def event2896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67079⟩⟩) 0 ⟨67078⟩ 2895

def event2897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67079⟩⟩) 1 ⟨6870⟩ 623

def event2898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67079⟩⟩) (.product (.predecessor 0 2896 .coefficient) (.predecessor 1 2897 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67079⟩⟩, .operator (⟨2895, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩)

def exact2900RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact2900RawTermsValid :
    exact2900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67079⟩⟩) exact2900RawTerms (.finite 226487908831958288795280) 2898 .exactZero (none)

def event2901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63218⟩⟩) 0 ⟨62865⟩ 2562

def event2902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63218⟩⟩) (.authority (.programFamilyFact))

def exact2903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩]

theorem exact2903RawTermsValid :
    exact2903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63218⟩⟩) exact2903RawTerms (.finite 22) 2902 .exactZero (none)

def event2904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63219⟩⟩) 0 ⟨63218⟩ 2903

def event2905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63219⟩⟩) 1 ⟨6732⟩ 633

def event2906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63219⟩⟩) (.product (.predecessor 0 2904 .coefficient) (.predecessor 1 2905 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63219⟩⟩, .operator (⟨2903, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩)

def exact2908RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩]

theorem exact2908RawTermsValid :
    exact2908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63219⟩⟩) exact2908RawTerms (.finite 224377773035387248837560) 2906 .exactZero (none)

def event2909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60238⟩⟩) 0 ⟨59885⟩ 2585

def event2910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60238⟩⟩) (.authority (.programFamilyFact))

def exact2911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩]

theorem exact2911RawTermsValid :
    exact2911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60238⟩⟩) exact2911RawTerms (.finite 18) 2910 .exactZero (none)

def event2912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60239⟩⟩) 0 ⟨60238⟩ 2911

def event2913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60239⟩⟩) 1 ⟨6736⟩ 643

def event2914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60239⟩⟩) (.product (.predecessor 0 2912 .coefficient) (.predecessor 1 2913 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60239⟩⟩, .operator (⟨2911, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩)

def exact2916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩]

theorem exact2916RawTermsValid :
    exact2916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60239⟩⟩) exact2916RawTerms (.finite 222230617312560576599880) 2914 .exactZero (none)

def event2917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57258⟩⟩) 0 ⟨56905⟩ 2608

def event2918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57258⟩⟩) (.authority (.programFamilyFact))

def exact2919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩]

theorem exact2919RawTermsValid :
    exact2919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57258⟩⟩) exact2919RawTerms (.finite 16) 2918 .exactZero (none)

def event2920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57259⟩⟩) 0 ⟨57258⟩ 2919

def event2921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57259⟩⟩) 1 ⟨6741⟩ 653

def event2922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57259⟩⟩) (.product (.predecessor 0 2920 .coefficient) (.predecessor 1 2921 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57259⟩⟩, .operator (⟨2919, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩)

def exact2924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩]

theorem exact2924RawTermsValid :
    exact2924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57259⟩⟩) exact2924RawTerms (.finite 220778129617707239497920) 2922 .exactZero (none)

def event2925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54278⟩⟩) 0 ⟨53925⟩ 2631

def event2926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54278⟩⟩) (.authority (.programFamilyFact))

def exact2927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩]

theorem exact2927RawTermsValid :
    exact2927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54278⟩⟩) exact2927RawTerms (.finite 12) 2926 .exactZero (none)

def event2928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54279⟩⟩) 0 ⟨54278⟩ 2927

def event2929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54279⟩⟩) 1 ⟨6757⟩ 663

def event2930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54279⟩⟩) (.product (.predecessor 0 2928 .coefficient) (.predecessor 1 2929 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2931 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54279⟩⟩, .operator (⟨2927, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩)

def exact2932RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩]

theorem exact2932RawTermsValid :
    exact2932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54279⟩⟩) exact2932RawTerms (.finite 216532396355828254122960) 2930 .exactZero (none)

def event2933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51298⟩⟩) 0 ⟨50945⟩ 2654

def event2934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51298⟩⟩) (.authority (.programFamilyFact))

def exact2935RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩]

theorem exact2935RawTermsValid :
    exact2935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51298⟩⟩) exact2935RawTerms (.finite 10) 2934 .exactZero (none)

def event2936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51299⟩⟩) 0 ⟨51298⟩ 2935

def event2937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51299⟩⟩) 1 ⟨6768⟩ 673

def event2938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51299⟩⟩) (.product (.predecessor 0 2936 .coefficient) (.predecessor 1 2937 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2939 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51299⟩⟩, .operator (⟨2935, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩)

def exact2940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩]

theorem exact2940RawTermsValid :
    exact2940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51299⟩⟩) exact2940RawTerms (.finite 213251602471649038151400) 2938 .exactZero (none)

def event2941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32234⟩⟩) 0 ⟨31885⟩ 2677

def event2942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32234⟩⟩) (.authority (.programFamilyFact))

def exact2943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩]

theorem exact2943RawTermsValid :
    exact2943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32234⟩⟩) exact2943RawTerms (.finite 6) 2942 .exactZero (none)

def event2944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32235⟩⟩) 0 ⟨32234⟩ 2943

def event2945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32235⟩⟩) 1 ⟨6794⟩ 683

def event2946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32235⟩⟩) (.product (.predecessor 0 2944 .coefficient) (.predecessor 1 2945 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2947 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32235⟩⟩, .operator (⟨2943, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩)

def exact2948RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩]

theorem exact2948RawTermsValid :
    exact2948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32235⟩⟩) exact2948RawTerms (.finite 201065796616126235971320) 2946 .exactZero (none)

def event2949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22214⟩⟩) 0 ⟨21865⟩ 2700

def event2950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22214⟩⟩) (.authority (.programFamilyFact))

def exact2951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩]

theorem exact2951RawTermsValid :
    exact2951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22214⟩⟩) exact2951RawTerms (.finite 4) 2950 .exactZero (none)

def event2952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22215⟩⟩) 0 ⟨22214⟩ 2951

def event2953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22215⟩⟩) 1 ⟨6822⟩ 693

def event2954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22215⟩⟩) (.product (.predecessor 0 2952 .coefficient) (.predecessor 1 2953 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22215⟩⟩, .operator (⟨2951, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩)

def exact2956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩]

theorem exact2956RawTermsValid :
    exact2956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22215⟩⟩) exact2956RawTerms (.finite 187661410175051153573232) 2954 .exactZero (none)

def event2957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18994⟩⟩) 0 ⟨18645⟩ 2723

def event2958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18994⟩⟩) (.authority (.programFamilyFact))

def exact2959RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩]

theorem exact2959RawTermsValid :
    exact2959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18994⟩⟩) exact2959RawTerms (.finite 3) 2958 .exactZero (none)

def event2960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18995⟩⟩) 0 ⟨18994⟩ 2959

def event2961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18995⟩⟩) 1 ⟨6846⟩ 703

def event2962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18995⟩⟩) (.product (.predecessor 0 2960 .coefficient) (.predecessor 1 2961 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2963 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18995⟩⟩, .operator (⟨2959, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩)

def exact2964RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩]

theorem exact2964RawTermsValid :
    exact2964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18995⟩⟩) exact2964RawTerms (.finite 175932572039110456474905) 2962 .exactZero (none)

def event2965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16142⟩⟩) 0 ⟨15845⟩ 2746

def event2966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16142⟩⟩) (.authority (.programFamilyFact))

def exact2967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2967RawTermsValid :
    exact2967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16142⟩⟩) exact2967RawTerms (.finite 2) 2966 .exactZero (none)

def event2968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16143⟩⟩) 0 ⟨16142⟩ 2967

def event2969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16143⟩⟩) 1 ⟨6863⟩ 713

def event2970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16143⟩⟩) (.product (.predecessor 0 2968 .coefficient) (.predecessor 1 2969 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event2971 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16143⟩⟩, .operator (⟨2967, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩)

def exact2972RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2972RawTermsValid :
    exact2972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16143⟩⟩) exact2972RawTerms (.finite 156384508479209294644360) 2970 .exactZero (none)

def event2973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16144⟩⟩) 0 ⟨6728⟩ 728

def event2974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16144⟩⟩) 1 ⟨16143⟩ 2972

def event2975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16144⟩⟩) (.sum [.predecessor 0 2973 .coefficient, .predecessor 1 2974 .coefficient])

def exact2976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2976RawTermsValid :
    exact2976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16144⟩⟩) exact2976RawTerms (.finite 156384508479209294644360) 2975 .exactZero (none)

def event2977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18996⟩⟩) 0 ⟨16144⟩ 2976

def event2978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18996⟩⟩) 1 ⟨18995⟩ 2964

def event2979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18996⟩⟩) (.sum [.predecessor 0 2977 .coefficient, .predecessor 1 2978 .coefficient])

def exact2980RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2980RawTermsValid :
    exact2980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18996⟩⟩) exact2980RawTerms (.finite 332317080518319751119265) 2979 .exactZero (none)

def event2981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22216⟩⟩) 0 ⟨18996⟩ 2980

def event2982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22216⟩⟩) 1 ⟨22215⟩ 2956

def event2983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22216⟩⟩) (.sum [.predecessor 0 2981 .coefficient, .predecessor 1 2982 .coefficient])

def exact2984RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2984RawTermsValid :
    exact2984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22216⟩⟩) exact2984RawTerms (.finite 519978490693370904692497) 2983 .exactZero (none)

def event2985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32236⟩⟩) 0 ⟨22216⟩ 2984

def event2986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32236⟩⟩) 1 ⟨32235⟩ 2948

def event2987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32236⟩⟩) (.sum [.predecessor 0 2985 .coefficient, .predecessor 1 2986 .coefficient])

def exact2988RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2988RawTermsValid :
    exact2988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2988 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32236⟩⟩) exact2988RawTerms (.finite 721044287309497140663817) 2987 .exactZero (none)

def event2989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51300⟩⟩) 0 ⟨32236⟩ 2988

def event2990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51300⟩⟩) 1 ⟨51299⟩ 2940

def event2991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51300⟩⟩) (.sum [.predecessor 0 2989 .coefficient, .predecessor 1 2990 .coefficient])

def exact2992RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2992RawTermsValid :
    exact2992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51300⟩⟩) exact2992RawTerms (.finite 934295889781146178815217) 2991 .exactZero (none)

def event2993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54280⟩⟩) 0 ⟨51300⟩ 2992

def event2994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54280⟩⟩) 1 ⟨54279⟩ 2932

def event2995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54280⟩⟩) (.sum [.predecessor 0 2993 .coefficient, .predecessor 1 2994 .coefficient])

def exact2996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact2996RawTermsValid :
    exact2996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event2996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54280⟩⟩) exact2996RawTerms (.finite 1150828286136974432938177) 2995 .exactZero (none)

def event2997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57260⟩⟩) 0 ⟨54280⟩ 2996

def event2998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57260⟩⟩) 1 ⟨57259⟩ 2924

def event2999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57260⟩⟩) (.sum [.predecessor 0 2997 .coefficient, .predecessor 1 2998 .coefficient])

def exact3000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact3000RawTermsValid :
    exact3000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57260⟩⟩) exact3000RawTerms (.finite 1371606415754681672436097) 2999 .exactZero (none)

def event3001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60240⟩⟩) 0 ⟨57260⟩ 3000

def event3002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60240⟩⟩) 1 ⟨60239⟩ 2916

def event3003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60240⟩⟩) (.sum [.predecessor 0 3001 .coefficient, .predecessor 1 3002 .coefficient])

def exact3004RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact3004RawTermsValid :
    exact3004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60240⟩⟩) exact3004RawTerms (.finite 1593837033067242249035977) 3003 .exactZero (none)

def event3005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63220⟩⟩) 0 ⟨60240⟩ 3004

def event3006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63220⟩⟩) 1 ⟨63219⟩ 2908

def event3007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63220⟩⟩) (.sum [.predecessor 0 3005 .coefficient, .predecessor 1 3006 .coefficient])

def exact3008RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩]

theorem exact3008RawTermsValid :
    exact3008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3008 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63220⟩⟩) exact3008RawTerms (.finite 1818214806102629497873537) 3007 .exactZero (none)

def event3009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67080⟩⟩) 0 ⟨63220⟩ 3008

def event3010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67080⟩⟩) 1 ⟨67079⟩ 2900

def event3011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67080⟩⟩) (.sum [.predecessor 0 3009 .coefficient, .predecessor 1 3010 .coefficient])

def exact3012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3012RawTermsValid :
    exact3012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67080⟩⟩) exact3012RawTerms (.finite 2044702714934587786668817) 3011 .exactZero (none)

def event3013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67081⟩⟩) 0 ⟨67080⟩ 3012

def event3014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67081⟩⟩) 1 ⟨26714⟩ 2892

def event3015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67081⟩⟩) (.sum [.predecessor 0 3013 .coefficient, .predecessor 1 3014 .coefficient])

def exact3016RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3016RawTermsValid :
    exact3016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67081⟩⟩) exact3016RawTerms (.finite 2271712485307633536959017) 3015 .exactZero (none)

def event3017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67082⟩⟩) 0 ⟨67081⟩ 3016

def event3018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67082⟩⟩) 1 ⟨29394⟩ 2884

def event3019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67082⟩⟩) (.sum [.predecessor 0 3017 .coefficient, .predecessor 1 3018 .coefficient])

def exact3020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3020RawTermsValid :
    exact3020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67082⟩⟩) exact3020RawTerms (.finite 2499949335520533588602137) 3019 .exactZero (none)

def event3021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67083⟩⟩) 0 ⟨67082⟩ 3020

def event3022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67083⟩⟩) 1 ⟨35051⟩ 2876

def event3023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67083⟩⟩) (.sum [.predecessor 0 3021 .coefficient, .predecessor 1 3022 .coefficient])

def exact3024RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3024RawTermsValid :
    exact3024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67083⟩⟩) exact3024RawTerms (.finite 2728804713782791092959737) 3023 .exactZero (none)

def event3025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67084⟩⟩) 0 ⟨67083⟩ 3024

def event3026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67084⟩⟩) 1 ⟨37731⟩ 2868

def event3027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67084⟩⟩) (.sum [.predecessor 0 3025 .coefficient, .predecessor 1 3026 .coefficient])

def exact3028RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3028RawTermsValid :
    exact3028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67084⟩⟩) exact3028RawTerms (.finite 2957926202950004710694497) 3027 .exactZero (none)

def event3029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67085⟩⟩) 0 ⟨67084⟩ 3028

def event3030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67085⟩⟩) 1 ⟨40414⟩ 2860

def event3031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67085⟩⟩) (.sum [.predecessor 0 3029 .coefficient, .predecessor 1 3030 .coefficient])

def exact3032RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3032RawTermsValid :
    exact3032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67085⟩⟩) exact3032RawTerms (.finite 3187511970717354526236217) 3031 .exactZero (none)

def event3033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67086⟩⟩) 0 ⟨67085⟩ 3032

def event3034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67086⟩⟩) 1 ⟨43094⟩ 2852

def event3035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67086⟩⟩) (.sum [.predecessor 0 3033 .coefficient, .predecessor 1 3034 .coefficient])

def exact3036RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3036RawTermsValid :
    exact3036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67086⟩⟩) exact3036RawTerms (.finite 3417662756781096507033577) 3035 .exactZero (none)

def event3037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67087⟩⟩) 0 ⟨67086⟩ 3036

def event3038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67087⟩⟩) 1 ⟨45771⟩ 2844

def event3039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67087⟩⟩) (.sum [.predecessor 0 3037 .coefficient, .predecessor 1 3038 .coefficient])

def exact3040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3040RawTermsValid :
    exact3040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67087⟩⟩) exact3040RawTerms (.finite 3648263642165693263543057) 3039 .exactZero (none)

def event3041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67088⟩⟩) 0 ⟨67087⟩ 3040

def event3042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67088⟩⟩) 1 ⟨48451⟩ 2836

def event3043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67088⟩⟩) (.sum [.predecessor 0 3041 .coefficient, .predecessor 1 3042 .coefficient])

def exact3044RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3044RawTermsValid :
    exact3044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67088⟩⟩) exact3044RawTerms (.finite 3878994884184198780231457) 3043 .exactZero (none)

def event3045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67609⟩⟩) 0 ⟨67088⟩ 3044

def event3046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67609⟩⟩) 1 ⟨67607⟩ 2828

def event3047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67609⟩⟩) (.sum [.predecessor 0 3045 .coefficient, .predecessor 1 3046 .coefficient])

def exact3048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3048RawTermsValid :
    exact3048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67609⟩⟩) exact3048RawTerms (.finite 8101376613122849735629177) 3047 .exactZero (none)

def event3049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67610⟩⟩) 0 ⟨67609⟩ 3048

def event3050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67610⟩⟩) 1 ⟨6779⟩ 2325

def event3051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67610⟩⟩) (.product (.predecessor 0 3049 .coefficient) (.predecessor 1 3050 .coefficient) (⟨false, true, none, none, some 1⟩))

def event3052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 5⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (-1)⟩)

def event3053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 7⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩)

def event3054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 8⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩)

def event3055 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 9⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩)

def event3056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 11⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩)

def event3057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 12⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩)

def event3058 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 13⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩)

def event3059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 15⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩)

def event3060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 16⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩)

def event3061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 18⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩)

def event3062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 0⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩)

def event3063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 1⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩)

def event3064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 2⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩)

def event3065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 3⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩)

def event3066 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 4⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩)

def event3067 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 6⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩)

def event3068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 10⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩)

def event3069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 14⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩)

def event3070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67610⟩⟩, .operator (⟨3048, 17⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩)

def exact3071RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨67606⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45770⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43093⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40413⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37730⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩, (1)⟩]

theorem exact3071RawTermsValid :
    exact3071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3071 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67610⟩⟩) exact3071RawTerms (.finite 12779664183506592515459042646047787215847583510316161333366657797817444811940860682592876895071820633746164080432004147536360702356640042135442291460260264826650336241668467965270988328085627535360) 3051 .exactZero (none)

def eventLeaf176 : Array AnnotatedEvent := #[
  { event := event2816
    frameStart := 0 },
  { event := event2817
    frameStart := 0 },
  { event := event2818
    frameStart := 0 },
  { event := event2819
    frameStart := 0 },
  { event := event2820
    frameStart := 0 },
  { event := event2821
    frameStart := 0 },
  { event := event2822
    frameStart := 0 },
  { event := event2823
    frameStart := 0 },
  { event := event2824
    frameStart := 0 },
  { event := event2825
    frameStart := 0 },
  { event := event2826
    frameStart := 0 },
  { event := event2827
    frameStart := 0 },
  { event := event2828
    frameStart := 0 },
  { event := event2829
    frameStart := 0 },
  { event := event2830
    frameStart := 0 },
  { event := event2831
    frameStart := 0 }
]

def eventLeaf177 : Array AnnotatedEvent := #[
  { event := event2832
    frameStart := 0 },
  { event := event2833
    frameStart := 0 },
  { event := event2834
    frameStart := 0 },
  { event := event2835
    frameStart := 0 },
  { event := event2836
    frameStart := 0 },
  { event := event2837
    frameStart := 0 },
  { event := event2838
    frameStart := 0 },
  { event := event2839
    frameStart := 0 },
  { event := event2840
    frameStart := 0 },
  { event := event2841
    frameStart := 0 },
  { event := event2842
    frameStart := 0 },
  { event := event2843
    frameStart := 0 },
  { event := event2844
    frameStart := 0 },
  { event := event2845
    frameStart := 0 },
  { event := event2846
    frameStart := 0 },
  { event := event2847
    frameStart := 0 }
]

def eventLeaf178 : Array AnnotatedEvent := #[
  { event := event2848
    frameStart := 0 },
  { event := event2849
    frameStart := 0 },
  { event := event2850
    frameStart := 0 },
  { event := event2851
    frameStart := 0 },
  { event := event2852
    frameStart := 0 },
  { event := event2853
    frameStart := 0 },
  { event := event2854
    frameStart := 0 },
  { event := event2855
    frameStart := 0 },
  { event := event2856
    frameStart := 0 },
  { event := event2857
    frameStart := 0 },
  { event := event2858
    frameStart := 0 },
  { event := event2859
    frameStart := 0 },
  { event := event2860
    frameStart := 0 },
  { event := event2861
    frameStart := 0 },
  { event := event2862
    frameStart := 0 },
  { event := event2863
    frameStart := 0 }
]

def eventLeaf179 : Array AnnotatedEvent := #[
  { event := event2864
    frameStart := 0 },
  { event := event2865
    frameStart := 0 },
  { event := event2866
    frameStart := 0 },
  { event := event2867
    frameStart := 0 },
  { event := event2868
    frameStart := 0 },
  { event := event2869
    frameStart := 0 },
  { event := event2870
    frameStart := 0 },
  { event := event2871
    frameStart := 0 },
  { event := event2872
    frameStart := 0 },
  { event := event2873
    frameStart := 0 },
  { event := event2874
    frameStart := 0 },
  { event := event2875
    frameStart := 0 },
  { event := event2876
    frameStart := 0 },
  { event := event2877
    frameStart := 0 },
  { event := event2878
    frameStart := 0 },
  { event := event2879
    frameStart := 0 }
]

def eventLeaf180 : Array AnnotatedEvent := #[
  { event := event2880
    frameStart := 0 },
  { event := event2881
    frameStart := 0 },
  { event := event2882
    frameStart := 0 },
  { event := event2883
    frameStart := 0 },
  { event := event2884
    frameStart := 0 },
  { event := event2885
    frameStart := 0 },
  { event := event2886
    frameStart := 0 },
  { event := event2887
    frameStart := 0 },
  { event := event2888
    frameStart := 0 },
  { event := event2889
    frameStart := 0 },
  { event := event2890
    frameStart := 0 },
  { event := event2891
    frameStart := 0 },
  { event := event2892
    frameStart := 0 },
  { event := event2893
    frameStart := 0 },
  { event := event2894
    frameStart := 0 },
  { event := event2895
    frameStart := 0 }
]

def eventLeaf181 : Array AnnotatedEvent := #[
  { event := event2896
    frameStart := 0 },
  { event := event2897
    frameStart := 0 },
  { event := event2898
    frameStart := 0 },
  { event := event2899
    frameStart := 0 },
  { event := event2900
    frameStart := 0 },
  { event := event2901
    frameStart := 0 },
  { event := event2902
    frameStart := 0 },
  { event := event2903
    frameStart := 0 },
  { event := event2904
    frameStart := 0 },
  { event := event2905
    frameStart := 0 },
  { event := event2906
    frameStart := 0 },
  { event := event2907
    frameStart := 0 },
  { event := event2908
    frameStart := 0 },
  { event := event2909
    frameStart := 0 },
  { event := event2910
    frameStart := 0 },
  { event := event2911
    frameStart := 0 }
]

def eventLeaf182 : Array AnnotatedEvent := #[
  { event := event2912
    frameStart := 0 },
  { event := event2913
    frameStart := 0 },
  { event := event2914
    frameStart := 0 },
  { event := event2915
    frameStart := 0 },
  { event := event2916
    frameStart := 0 },
  { event := event2917
    frameStart := 0 },
  { event := event2918
    frameStart := 0 },
  { event := event2919
    frameStart := 0 },
  { event := event2920
    frameStart := 0 },
  { event := event2921
    frameStart := 0 },
  { event := event2922
    frameStart := 0 },
  { event := event2923
    frameStart := 0 },
  { event := event2924
    frameStart := 0 },
  { event := event2925
    frameStart := 0 },
  { event := event2926
    frameStart := 0 },
  { event := event2927
    frameStart := 0 }
]

def eventLeaf183 : Array AnnotatedEvent := #[
  { event := event2928
    frameStart := 0 },
  { event := event2929
    frameStart := 0 },
  { event := event2930
    frameStart := 0 },
  { event := event2931
    frameStart := 0 },
  { event := event2932
    frameStart := 0 },
  { event := event2933
    frameStart := 0 },
  { event := event2934
    frameStart := 0 },
  { event := event2935
    frameStart := 0 },
  { event := event2936
    frameStart := 0 },
  { event := event2937
    frameStart := 0 },
  { event := event2938
    frameStart := 0 },
  { event := event2939
    frameStart := 0 },
  { event := event2940
    frameStart := 0 },
  { event := event2941
    frameStart := 0 },
  { event := event2942
    frameStart := 0 },
  { event := event2943
    frameStart := 0 }
]

def eventLeaf184 : Array AnnotatedEvent := #[
  { event := event2944
    frameStart := 0 },
  { event := event2945
    frameStart := 0 },
  { event := event2946
    frameStart := 0 },
  { event := event2947
    frameStart := 0 },
  { event := event2948
    frameStart := 0 },
  { event := event2949
    frameStart := 0 },
  { event := event2950
    frameStart := 0 },
  { event := event2951
    frameStart := 0 },
  { event := event2952
    frameStart := 0 },
  { event := event2953
    frameStart := 0 },
  { event := event2954
    frameStart := 0 },
  { event := event2955
    frameStart := 0 },
  { event := event2956
    frameStart := 0 },
  { event := event2957
    frameStart := 0 },
  { event := event2958
    frameStart := 0 },
  { event := event2959
    frameStart := 0 }
]

def eventLeaf185 : Array AnnotatedEvent := #[
  { event := event2960
    frameStart := 0 },
  { event := event2961
    frameStart := 0 },
  { event := event2962
    frameStart := 0 },
  { event := event2963
    frameStart := 0 },
  { event := event2964
    frameStart := 0 },
  { event := event2965
    frameStart := 0 },
  { event := event2966
    frameStart := 0 },
  { event := event2967
    frameStart := 0 },
  { event := event2968
    frameStart := 0 },
  { event := event2969
    frameStart := 0 },
  { event := event2970
    frameStart := 0 },
  { event := event2971
    frameStart := 0 },
  { event := event2972
    frameStart := 0 },
  { event := event2973
    frameStart := 0 },
  { event := event2974
    frameStart := 0 },
  { event := event2975
    frameStart := 0 }
]

def eventLeaf186 : Array AnnotatedEvent := #[
  { event := event2976
    frameStart := 0 },
  { event := event2977
    frameStart := 0 },
  { event := event2978
    frameStart := 0 },
  { event := event2979
    frameStart := 0 },
  { event := event2980
    frameStart := 0 },
  { event := event2981
    frameStart := 0 },
  { event := event2982
    frameStart := 0 },
  { event := event2983
    frameStart := 0 },
  { event := event2984
    frameStart := 0 },
  { event := event2985
    frameStart := 0 },
  { event := event2986
    frameStart := 0 },
  { event := event2987
    frameStart := 0 },
  { event := event2988
    frameStart := 0 },
  { event := event2989
    frameStart := 0 },
  { event := event2990
    frameStart := 0 },
  { event := event2991
    frameStart := 0 }
]

def eventLeaf187 : Array AnnotatedEvent := #[
  { event := event2992
    frameStart := 0 },
  { event := event2993
    frameStart := 0 },
  { event := event2994
    frameStart := 0 },
  { event := event2995
    frameStart := 0 },
  { event := event2996
    frameStart := 0 },
  { event := event2997
    frameStart := 0 },
  { event := event2998
    frameStart := 0 },
  { event := event2999
    frameStart := 0 },
  { event := event3000
    frameStart := 0 },
  { event := event3001
    frameStart := 0 },
  { event := event3002
    frameStart := 0 },
  { event := event3003
    frameStart := 0 },
  { event := event3004
    frameStart := 0 },
  { event := event3005
    frameStart := 0 },
  { event := event3006
    frameStart := 0 },
  { event := event3007
    frameStart := 0 }
]

def eventLeaf188 : Array AnnotatedEvent := #[
  { event := event3008
    frameStart := 0 },
  { event := event3009
    frameStart := 0 },
  { event := event3010
    frameStart := 0 },
  { event := event3011
    frameStart := 0 },
  { event := event3012
    frameStart := 0 },
  { event := event3013
    frameStart := 0 },
  { event := event3014
    frameStart := 0 },
  { event := event3015
    frameStart := 0 },
  { event := event3016
    frameStart := 0 },
  { event := event3017
    frameStart := 0 },
  { event := event3018
    frameStart := 0 },
  { event := event3019
    frameStart := 0 },
  { event := event3020
    frameStart := 0 },
  { event := event3021
    frameStart := 0 },
  { event := event3022
    frameStart := 0 },
  { event := event3023
    frameStart := 0 }
]

def eventLeaf189 : Array AnnotatedEvent := #[
  { event := event3024
    frameStart := 0 },
  { event := event3025
    frameStart := 0 },
  { event := event3026
    frameStart := 0 },
  { event := event3027
    frameStart := 0 },
  { event := event3028
    frameStart := 0 },
  { event := event3029
    frameStart := 0 },
  { event := event3030
    frameStart := 0 },
  { event := event3031
    frameStart := 0 },
  { event := event3032
    frameStart := 0 },
  { event := event3033
    frameStart := 0 },
  { event := event3034
    frameStart := 0 },
  { event := event3035
    frameStart := 0 },
  { event := event3036
    frameStart := 0 },
  { event := event3037
    frameStart := 0 },
  { event := event3038
    frameStart := 0 },
  { event := event3039
    frameStart := 0 }
]

def eventLeaf190 : Array AnnotatedEvent := #[
  { event := event3040
    frameStart := 0 },
  { event := event3041
    frameStart := 0 },
  { event := event3042
    frameStart := 0 },
  { event := event3043
    frameStart := 0 },
  { event := event3044
    frameStart := 0 },
  { event := event3045
    frameStart := 0 },
  { event := event3046
    frameStart := 0 },
  { event := event3047
    frameStart := 0 },
  { event := event3048
    frameStart := 0 },
  { event := event3049
    frameStart := 0 },
  { event := event3050
    frameStart := 0 },
  { event := event3051
    frameStart := 0 },
  { event := event3052
    frameStart := 0 },
  { event := event3053
    frameStart := 0 },
  { event := event3054
    frameStart := 0 },
  { event := event3055
    frameStart := 0 }
]

def eventLeaf191 : Array AnnotatedEvent := #[
  { event := event3056
    frameStart := 0 },
  { event := event3057
    frameStart := 0 },
  { event := event3058
    frameStart := 0 },
  { event := event3059
    frameStart := 0 },
  { event := event3060
    frameStart := 0 },
  { event := event3061
    frameStart := 0 },
  { event := event3062
    frameStart := 0 },
  { event := event3063
    frameStart := 0 },
  { event := event3064
    frameStart := 0 },
  { event := event3065
    frameStart := 0 },
  { event := event3066
    frameStart := 0 },
  { event := event3067
    frameStart := 0 },
  { event := event3068
    frameStart := 0 },
  { event := event3069
    frameStart := 0 },
  { event := event3070
    frameStart := 0 },
  { event := event3071
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events011
