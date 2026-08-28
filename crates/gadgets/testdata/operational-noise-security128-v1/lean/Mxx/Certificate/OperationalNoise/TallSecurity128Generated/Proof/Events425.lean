import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events425

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event108800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27932⟩⟩) 1 ⟨27931⟩ 108612

def event108801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27932⟩⟩) (.sum [.predecessor 0 108799 .coefficient, .predecessor 1 108800 .coefficient])

def event108802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27932⟩⟩, .operator (⟨108798, 2⟩, ⟨108612, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨27415⟩⟩]⟩, (-1)⟩)

def event108803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27932⟩⟩, .operator (⟨108798, 1⟩, ⟨108612, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩, (1)⟩)

def event108804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27932⟩⟩) (.sum [.result 108798 .summary, .result 108612 .summary])

def exact108805RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact108805RawTermsValid :
    exact108805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27932⟩⟩) exact108805RawTerms .large 108801 (.finite 2998072422921948889088) (some (108804))

def event108806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28316⟩⟩) 0 ⟨27932⟩ 108805

def event108807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28316⟩⟩) 1 ⟨28314⟩ 108528

def event108808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28316⟩⟩) (.product (.predecessor 0 108806 .coefficient) (.predecessor 1 108807 .coefficient) (⟨false, false, none, none, none⟩))

def event108809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28316⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) [⟨.result 108528 .coefficient, false, none⟩])

def event108810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28316⟩⟩) (.product (.result 108805 .summary) (.transfer 108809) (⟨false, false, none, none, none⟩))

def event108811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28316⟩⟩, .operator (⟨108805, 0⟩, ⟨108528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩)

def event108812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28316⟩⟩, .operator (⟨108805, 1⟩, ⟨108528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩)

def event108813 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28316⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28314⟩⟩) ⟨27570⟩ 108525)

def event108814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28316⟩⟩, .relation 108813 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (-1)⟩)

def exact108815RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (-1)⟩]

theorem exact108815RawTermsValid :
    exact108815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28316⟩⟩) exact108815RawTerms .large 108808 (.finite 32191557518723128098041228165120) (some (108810))

def event108816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27176⟩⟩) 0 ⟨26417⟩ 4760

def event108817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27176⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact108818RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩]

theorem exact108818RawTermsValid :
    exact108818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27176⟩⟩) exact108818RawTerms (.finite 5647228698) 108817 .exactZero (none)

def event108819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27178⟩⟩) 0 ⟨27176⟩ 108818

def event108820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27178⟩⟩) 1 ⟨2370⟩ 4

def event108821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27178⟩⟩) (.scale (.predecessor 0 108819 .coefficient) (.value (.predecessor 1 108820 .coefficient)))

def exact108822RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩]

theorem exact108822RawTermsValid :
    exact108822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27178⟩⟩) exact108822RawTerms (.finite 5647228698) 108821 .exactZero (none)

def event108823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27179⟩⟩) 0 ⟨5770⟩ 105245

def event108824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27179⟩⟩) 1 ⟨27178⟩ 108822

def event108825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27179⟩⟩) (.product (.predecessor 0 108823 .coefficient) (.predecessor 1 108824 .coefficient) (⟨false, false, none, none, none⟩))

def event108826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27179⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) [⟨.result 108818 .coefficient, false, none⟩])

def event108827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27179⟩⟩) (.product (.result 105245 .summary) (.transfer 108826) (⟨false, false, none, none, none⟩))

def event108828 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27179⟩⟩, .operator (⟨105245, 0⟩, ⟨108822, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩)

def event108829 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27177⟩⟩)

def event108830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event108831 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event108832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event108833 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event108834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event108835 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event108836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event108837 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event108838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 108837

def event108839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 108835

def event108840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 108838 .coefficient) (.value (.predecessor 1 108839 .coefficient)))

def event108841 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event108842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 108841

def event108843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 108833

def event108844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 108842 .coefficient, .predecessor 1 108843 .coefficient])

def event108845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event108846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 108845

def event108847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 108831

def event108848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 108847 .coefficient))

def event108849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event108850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26118⟩⟩) 0 ⟨5766⟩ 108849

def event108851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26118⟩⟩) (.authority (.programFamilyFact))

def exact108852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26118⟩⟩], []⟩, (1)⟩]

theorem exact108852RawTermsValid :
    exact108852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26118⟩⟩) exact108852RawTerms (.finite 30) 108851 .exactZero (none)

def event108853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12996⟩⟩) 0 ⟨5766⟩ 108849

def event108854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12996⟩⟩) (.authority (.programFamilyFact))

def exact108855RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩], []⟩, (1)⟩]

theorem exact108855RawTermsValid :
    exact108855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12996⟩⟩) exact108855RawTerms (.finite 30) 108854 .exactZero (none)

def event108856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26119⟩⟩) 0 ⟨12996⟩ 108855

def event108857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26119⟩⟩) 1 ⟨26118⟩ 108852

def event108858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26119⟩⟩) (.product (.predecessor 0 108856 .coefficient) (.predecessor 1 108857 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event108859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26119⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], []⟩) [⟨.result 108855 .coefficient, true, some 1⟩, ⟨.result 108852 .coefficient, true, some 1⟩])

def event108860 : Event := .survivorFold (1) 108859

def exact108861RawTerms : List Term := []

theorem exact108861RawTermsValid :
    exact108861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26119⟩⟩) exact108861RawTerms (.finite 900) 108858 (.finite 900) (some (108859))

def event108862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26120⟩⟩) 0 ⟨26119⟩ 108861

def event108863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26120⟩⟩) (.identity (.predecessor 0 108862 .coefficient))

def event108864 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26120⟩⟩) (.finite 900)

def event108865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26416⟩⟩) 0 ⟨26120⟩ 108864

def event108866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26416⟩⟩) (.authority (.programFamilyFact))

def exact108867RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], []⟩, (1)⟩]

theorem exact108867RawTermsValid :
    exact108867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26416⟩⟩) exact108867RawTerms (.finite 30) 108866 .exactZero (none)

def event108868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26417⟩⟩) 0 ⟨26416⟩ 108867

def event108869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26417⟩⟩) (.identity (.predecessor 0 108868 .coefficient))

def event108870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26417⟩⟩) (.finite 30)

def event108871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27176⟩⟩) 0 ⟨26417⟩ 108870

def event108872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27176⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact108873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩]

theorem exact108873RawTermsValid :
    exact108873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27176⟩⟩) exact108873RawTerms (.finite 5647228698) 108872 .exactZero (none)

def event108874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact108875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact108875RawTermsValid :
    exact108875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact108875RawTerms .large 108874 .exactZero (none)

def event108876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27177⟩⟩) 0 ⟨35⟩ 108875

def event108877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27177⟩⟩) 1 ⟨27176⟩ 108873

def event108878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27177⟩⟩) (.product (.predecessor 0 108876 .coefficient) (.predecessor 1 108877 .coefficient) (⟨false, false, none, none, none⟩))

def event108879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27177⟩⟩, .operator (⟨108875, 0⟩, ⟨108873, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩)

def exact108880RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩]

theorem exact108880RawTermsValid :
    exact108880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27177⟩⟩) exact108880RawTerms .large 108878 .exactZero (none)

def event108881 : Event := .preFoldPolynomial 108880 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩] .exactZero none

def exact108882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩, (1)⟩]

def event108882 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27177⟩⟩) 108881 exact108882RawTerms .large 108878 .exactZero (none)

def event108883 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28318⟩⟩)

def event108884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event108885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event108886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event108887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event108888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event108889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event108890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event108891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event108892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 108891

def event108893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 108889

def event108894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 108892 .coefficient) (.value (.predecessor 1 108893 .coefficient)))

def event108895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event108896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 108895

def event108897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 108887

def event108898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 108896 .coefficient, .predecessor 1 108897 .coefficient])

def event108899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event108900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 108899

def event108901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 108885

def event108902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 108901 .coefficient))

def event108903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event108904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26118⟩⟩) 0 ⟨5766⟩ 108903

def event108905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26118⟩⟩) (.authority (.programFamilyFact))

def exact108906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26118⟩⟩], []⟩, (1)⟩]

theorem exact108906RawTermsValid :
    exact108906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26118⟩⟩) exact108906RawTerms (.finite 30) 108905 .exactZero (none)

def event108907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12996⟩⟩) 0 ⟨5766⟩ 108903

def event108908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12996⟩⟩) (.authority (.programFamilyFact))

def exact108909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩], []⟩, (1)⟩]

theorem exact108909RawTermsValid :
    exact108909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12996⟩⟩) exact108909RawTerms (.finite 30) 108908 .exactZero (none)

def event108910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26119⟩⟩) 0 ⟨12996⟩ 108909

def event108911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26119⟩⟩) 1 ⟨26118⟩ 108906

def event108912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26119⟩⟩) (.product (.predecessor 0 108910 .coefficient) (.predecessor 1 108911 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event108913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26119⟩⟩, .operator (⟨108909, 0⟩, ⟨108906, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], []⟩, (1)⟩)

def exact108914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], []⟩, (1)⟩]

theorem exact108914RawTermsValid :
    exact108914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26119⟩⟩) exact108914RawTerms (.finite 900) 108912 .exactZero (none)

def event108915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26120⟩⟩) 0 ⟨26119⟩ 108914

def event108916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26120⟩⟩) (.identity (.predecessor 0 108915 .coefficient))

def event108917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26120⟩⟩) (.finite 900)

def event108918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26416⟩⟩) 0 ⟨26120⟩ 108917

def event108919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26416⟩⟩) (.authority (.programFamilyFact))

def exact108920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], []⟩, (1)⟩]

theorem exact108920RawTermsValid :
    exact108920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26416⟩⟩) exact108920RawTerms (.finite 30) 108919 .exactZero (none)

def event108921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26417⟩⟩) 0 ⟨26416⟩ 108920

def event108922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26417⟩⟩) (.identity (.predecessor 0 108921 .coefficient))

def event108923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26417⟩⟩) (.finite 30)

def event108924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27568⟩⟩) 0 ⟨26417⟩ 108923

def event108925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27568⟩⟩) (.authority (.programFamilyFact))

def event108926 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27568⟩⟩) (.finite 3720)

def event108927 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event108928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27570⟩⟩) 0 ⟨7177⟩ 108927

def event108929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27570⟩⟩) 1 ⟨27568⟩ 108926

def event108930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27570⟩⟩) (.authority (.operator))

def exact108931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩]

theorem exact108931RawTermsValid :
    exact108931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27570⟩⟩) exact108931RawTerms .large 108930 .exactZero (none)

def event108932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28314⟩⟩) 0 ⟨27570⟩ 108931

def event108933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28314⟩⟩) (.authority (.operator))

def exact108934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩]

theorem exact108934RawTermsValid :
    exact108934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28314⟩⟩) exact108934RawTerms (.finite 8192) 108933 .exactZero (none)

def event108935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event108936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event108937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27770⟩⟩) 0 ⟨26417⟩ 108923

def event108938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27770⟩⟩) 1 ⟨136⟩ 108936

def event108939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27770⟩⟩) (.sum [.predecessor 0 108937 .coefficient, .predecessor 1 108938 .coefficient])

def event108940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27770⟩⟩) (.finite 30)

def event108941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27771⟩⟩) 0 ⟨27770⟩ 108940

def event108942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27771⟩⟩) (.identity (.predecessor 0 108941 .coefficient))

def exact108943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], []⟩, (1)⟩]

theorem exact108943RawTermsValid :
    exact108943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27771⟩⟩) exact108943RawTerms (.finite 30) 108942 .exactZero (none)

def event108944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact108945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact108945RawTermsValid :
    exact108945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact108945RawTerms .large 108944 .exactZero (none)

def event108946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27772⟩⟩) 0 ⟨6908⟩ 108945

def event108947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27772⟩⟩) 1 ⟨27771⟩ 108943

def event108948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27772⟩⟩) (.product (.predecessor 0 108946 .coefficient) (.predecessor 1 108947 .coefficient) (⟨false, false, none, none, none⟩))

def event108949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27772⟩⟩, .operator (⟨108945, 0⟩, ⟨108943, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact108950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact108950RawTermsValid :
    exact108950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27772⟩⟩) exact108950RawTerms .large 108948 .exactZero (none)

def event108951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 108927

def event108952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact108953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact108953RawTermsValid :
    exact108953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact108953RawTerms .large 108952 .exactZero (none)

def event108954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27773⟩⟩) 0 ⟨7189⟩ 108953

def event108955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27773⟩⟩) 1 ⟨27772⟩ 108950

def event108956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27773⟩⟩) (.sum [.predecessor 0 108954 .coefficient, .predecessor 1 108955 .coefficient])

def exact108957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact108957RawTermsValid :
    exact108957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27773⟩⟩) exact108957RawTerms .large 108956 .exactZero (none)

def event108958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28315⟩⟩) 0 ⟨27773⟩ 108957

def event108959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28315⟩⟩) 1 ⟨28314⟩ 108934

def event108960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28315⟩⟩) (.product (.predecessor 0 108958 .coefficient) (.predecessor 1 108959 .coefficient) (⟨false, false, none, none, none⟩))

def event108961 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28315⟩⟩, .operator (⟨108957, 0⟩, ⟨108934, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩)

def event108962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28315⟩⟩, .operator (⟨108957, 1⟩, ⟨108934, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩)

def event108963 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28315⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28314⟩⟩) ⟨27570⟩ 108931)

def event108964 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28315⟩⟩, .relation 108963 0, ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (-1)⟩)

def exact108965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (-1)⟩]

theorem exact108965RawTermsValid :
    exact108965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28315⟩⟩) exact108965RawTerms .large 108960 .exactZero (none)

def event108966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26632⟩⟩) 0 ⟨26417⟩ 108923

def event108967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26632⟩⟩) (.authority (.programFamilyFact))

def exact108968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], []⟩, (1)⟩]

theorem exact108968RawTermsValid :
    exact108968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26632⟩⟩) exact108968RawTerms (.finite 62) 108967 .exactZero (none)

def event108969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26633⟩⟩) 0 ⟨6908⟩ 108945

def event108970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26633⟩⟩) 1 ⟨26632⟩ 108968

def event108971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26633⟩⟩) (.product (.predecessor 0 108969 .coefficient) (.predecessor 1 108970 .coefficient) (⟨false, true, none, none, some 1⟩))

def event108972 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26633⟩⟩, .operator (⟨108945, 0⟩, ⟨108968, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact108973RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact108973RawTermsValid :
    exact108973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26633⟩⟩) exact108973RawTerms .large 108971 .exactZero (none)

def event108974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 108927

def event108975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact108976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact108976RawTermsValid :
    exact108976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact108976RawTerms .large 108975 .exactZero (none)

def event108977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26634⟩⟩) 0 ⟨7218⟩ 108976

def event108978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26634⟩⟩) 1 ⟨26633⟩ 108973

def event108979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26634⟩⟩) (.sum [.predecessor 0 108977 .coefficient, .predecessor 1 108978 .coefficient])

def exact108980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact108980RawTermsValid :
    exact108980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26634⟩⟩) exact108980RawTerms .large 108979 .exactZero (none)

def event108981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28318⟩⟩) 0 ⟨26634⟩ 108980

def event108982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28318⟩⟩) 1 ⟨28315⟩ 108965

def event108983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28318⟩⟩) (.sum [.predecessor 0 108981 .coefficient, .predecessor 1 108982 .coefficient])

def exact108984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact108984RawTermsValid :
    exact108984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28318⟩⟩) exact108984RawTerms .large 108983 .exactZero (none)

def event108985 : Event := .preFoldPolynomial 108984 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact108986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event108986 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28318⟩⟩) 108985 exact108986RawTerms .large 108983 .exactZero (none)

def event108987 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26417⟩⟩) ⟨⟨97⟩, ⟨79⟩, ⟨135⟩⟩ ⟨108829, 108987⟩

def event108988 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27179⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (1) 0 2 (.universal 108987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (none) 108986)

def event108989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27179⟩⟩, .relation 108988 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩)

def event108990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27179⟩⟩, .relation 108988 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩)

def event108991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27179⟩⟩, .relation 108988 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩)

def event108992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27179⟩⟩, .relation 108988 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact108993RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact108993RawTermsValid :
    exact108993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event108993 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27179⟩⟩) exact108993RawTerms .large 108825 (.finite 202072841853861888) (some (108827))

def event108994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28317⟩⟩) 0 ⟨27179⟩ 108993

def event108995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28317⟩⟩) 1 ⟨28316⟩ 108815

def event108996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28317⟩⟩) (.sum [.predecessor 0 108994 .coefficient, .predecessor 1 108995 .coefficient])

def event108997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28317⟩⟩, .operator (⟨108993, 0⟩, ⟨108815, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩, (1)⟩)

def event108998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28317⟩⟩, .operator (⟨108993, 2⟩, ⟨108815, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩, (-1)⟩)

def event108999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28317⟩⟩) (.sum [.result 108993 .summary, .result 108815 .summary])

def exact109000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26632⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact109000RawTermsValid :
    exact109000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28317⟩⟩) exact109000RawTerms .large 108996 (.finite 32191557518723330170883082027008) (some (108999))

def event109001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68689⟩⟩) 0 ⟨65797⟩ 4783

def event109002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68689⟩⟩) (.authority (.programFamilyFact))

def event109003 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68689⟩⟩) (.finite 3720)

def event109004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68691⟩⟩) 0 ⟨7177⟩ 15500

def event109005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68691⟩⟩) 1 ⟨68689⟩ 109003

def event109006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68691⟩⟩) (.authority (.operator))

def exact109007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68691⟩⟩]⟩, (1)⟩]

theorem exact109007RawTermsValid :
    exact109007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68691⟩⟩) exact109007RawTerms .large 109006 .exactZero (none)

def event109008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70256⟩⟩) 0 ⟨68691⟩ 109007

def event109009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70256⟩⟩) (.authority (.operator))

def exact109010RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70256⟩⟩]⟩, (1)⟩]

theorem exact109010RawTermsValid :
    exact109010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70256⟩⟩) exact109010RawTerms (.finite 8192) 109009 .exactZero (none)

def event109011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68535⟩⟩) 0 ⟨65474⟩ 4777

def event109012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68535⟩⟩) (.authority (.programFamilyFact))

def event109013 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68535⟩⟩) (.finite 3720)

def event109014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68536⟩⟩) 0 ⟨7177⟩ 15500

def event109015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68536⟩⟩) 1 ⟨68535⟩ 109013

def event109016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68536⟩⟩) (.authority (.operator))

def exact109017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68536⟩⟩]⟩, (1)⟩]

theorem exact109017RawTermsValid :
    exact109017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68536⟩⟩) exact109017RawTerms .large 109016 .exactZero (none)

def event109018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69251⟩⟩) 0 ⟨68536⟩ 109017

def event109019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69251⟩⟩) (.authority (.operator))

def exact109020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69251⟩⟩]⟩, (1)⟩]

theorem exact109020RawTermsValid :
    exact109020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69251⟩⟩) exact109020RawTerms (.finite 8192) 109019 .exactZero (none)

def event109021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25743⟩⟩) 0 ⟨25742⟩ 4766

def event109022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25743⟩⟩) 1 ⟨6992⟩ 105153

def event109023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25743⟩⟩) (.tensor (.predecessor 0 109021 .coefficient) (.predecessor 1 109022 .coefficient) true false)

def event109024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25743⟩⟩, .operator (⟨4766, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact109025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact109025RawTermsValid :
    exact109025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25743⟩⟩) exact109025RawTerms .large 109023 .exactZero (none)

def event109026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8696⟩⟩) 0 ⟨5768⟩ 105023

def event109027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8696⟩⟩) 1 ⟨7276⟩ 21088

def event109028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8696⟩⟩) (.product (.predecessor 0 109026 .coefficient) (.predecessor 1 109027 .coefficient) (⟨false, false, none, none, none⟩))

def event109029 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8696⟩⟩, .operator (⟨105023, 0⟩, ⟨21088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact109030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact109030RawTermsValid :
    exact109030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8696⟩⟩) exact109030RawTerms .large 109028 .exactZero (none)

def event109031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25744⟩⟩) 0 ⟨8696⟩ 109030

def event109032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25744⟩⟩) 1 ⟨25743⟩ 109025

def event109033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25744⟩⟩) (.sum [.predecessor 0 109031 .coefficient, .predecessor 1 109032 .coefficient])

def exact109034RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact109034RawTermsValid :
    exact109034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25744⟩⟩) exact109034RawTerms .large 109033 .exactZero (none)

def event109035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25745⟩⟩) 0 ⟨25744⟩ 109034

def event109036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25745⟩⟩) 1 ⟨102⟩ 21080

def event109037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25745⟩⟩) (.sum [.predecessor 0 109035 .coefficient, .predecessor 1 109036 .coefficient])

def event109038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25745⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩) [⟨.result 21080 .coefficient, false, none⟩])

def event109039 : Event := .survivorFold (1) 109038

def exact109040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact109040RawTermsValid :
    exact109040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25745⟩⟩) exact109040RawTerms .large 109037 (.finite 26) (some (109038))

def event109041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65475⟩⟩) 0 ⟨25745⟩ 109040

def event109042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65475⟩⟩) 1 ⟨65472⟩ 4769

def event109043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65475⟩⟩) (.product (.predecessor 0 109041 .coefficient) (.predecessor 1 109042 .coefficient) (⟨false, true, none, none, some 1⟩))

def event109044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65475⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨65472⟩⟩], []⟩) [⟨.result 4769 .coefficient, true, some 1⟩])

def event109045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65475⟩⟩) (.product (.result 109040 .summary) (.transfer 109044) (⟨false, false, none, none, none⟩))

def event109046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65475⟩⟩, .operator (⟨109040, 1⟩, ⟨4769, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event109047 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65475⟩⟩, .operator (⟨109040, 0⟩, ⟨4769, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact109048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨25742⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact109048RawTermsValid :
    exact109048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65475⟩⟩) exact109048RawTerms .large 109043 (.finite 23855104) (some (109045))

def event109049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65476⟩⟩) 0 ⟨65472⟩ 4769

def event109050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65476⟩⟩) 1 ⟨6992⟩ 105153

def event109051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65476⟩⟩) (.tensor (.predecessor 0 109049 .coefficient) (.predecessor 1 109050 .coefficient) true false)

def event109052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65476⟩⟩, .operator (⟨4769, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact109053RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨65472⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact109053RawTermsValid :
    exact109053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event109053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65476⟩⟩) exact109053RawTerms .large 109051 .exactZero (none)

def event109054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8714⟩⟩) 0 ⟨5768⟩ 105023

def event109055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8714⟩⟩) 1 ⟨7294⟩ 21129

def eventLeaf6800 : Array AnnotatedEvent := #[
  { event := event108800
    frameStart := 0 },
  { event := event108801
    frameStart := 0 },
  { event := event108802
    frameStart := 0 },
  { event := event108803
    frameStart := 0 },
  { event := event108804
    frameStart := 0 },
  { event := event108805
    frameStart := 0 },
  { event := event108806
    frameStart := 0 },
  { event := event108807
    frameStart := 0 },
  { event := event108808
    frameStart := 0 },
  { event := event108809
    frameStart := 0 },
  { event := event108810
    frameStart := 0 },
  { event := event108811
    frameStart := 0 },
  { event := event108812
    frameStart := 0 },
  { event := event108813
    frameStart := 0 },
  { event := event108814
    frameStart := 0 },
  { event := event108815
    frameStart := 0 }
]

def eventLeaf6801 : Array AnnotatedEvent := #[
  { event := event108816
    frameStart := 0 },
  { event := event108817
    frameStart := 0 },
  { event := event108818
    frameStart := 0 },
  { event := event108819
    frameStart := 0 },
  { event := event108820
    frameStart := 0 },
  { event := event108821
    frameStart := 0 },
  { event := event108822
    frameStart := 0 },
  { event := event108823
    frameStart := 0 },
  { event := event108824
    frameStart := 0 },
  { event := event108825
    frameStart := 0 },
  { event := event108826
    frameStart := 0 },
  { event := event108827
    frameStart := 0 },
  { event := event108828
    frameStart := 0 },
  { event := event108829
    frameStart := 108829 },
  { event := event108830
    frameStart := 108829 },
  { event := event108831
    frameStart := 108829 }
]

def eventLeaf6802 : Array AnnotatedEvent := #[
  { event := event108832
    frameStart := 108829 },
  { event := event108833
    frameStart := 108829 },
  { event := event108834
    frameStart := 108829 },
  { event := event108835
    frameStart := 108829 },
  { event := event108836
    frameStart := 108829 },
  { event := event108837
    frameStart := 108829 },
  { event := event108838
    frameStart := 108829 },
  { event := event108839
    frameStart := 108829 },
  { event := event108840
    frameStart := 108829 },
  { event := event108841
    frameStart := 108829 },
  { event := event108842
    frameStart := 108829 },
  { event := event108843
    frameStart := 108829 },
  { event := event108844
    frameStart := 108829 },
  { event := event108845
    frameStart := 108829 },
  { event := event108846
    frameStart := 108829 },
  { event := event108847
    frameStart := 108829 }
]

def eventLeaf6803 : Array AnnotatedEvent := #[
  { event := event108848
    frameStart := 108829 },
  { event := event108849
    frameStart := 108829 },
  { event := event108850
    frameStart := 108829 },
  { event := event108851
    frameStart := 108829 },
  { event := event108852
    frameStart := 108829 },
  { event := event108853
    frameStart := 108829 },
  { event := event108854
    frameStart := 108829 },
  { event := event108855
    frameStart := 108829 },
  { event := event108856
    frameStart := 108829 },
  { event := event108857
    frameStart := 108829 },
  { event := event108858
    frameStart := 108829 },
  { event := event108859
    frameStart := 108829 },
  { event := event108860
    frameStart := 108829 },
  { event := event108861
    frameStart := 108829 },
  { event := event108862
    frameStart := 108829 },
  { event := event108863
    frameStart := 108829 }
]

def eventLeaf6804 : Array AnnotatedEvent := #[
  { event := event108864
    frameStart := 108829 },
  { event := event108865
    frameStart := 108829 },
  { event := event108866
    frameStart := 108829 },
  { event := event108867
    frameStart := 108829 },
  { event := event108868
    frameStart := 108829 },
  { event := event108869
    frameStart := 108829 },
  { event := event108870
    frameStart := 108829 },
  { event := event108871
    frameStart := 108829 },
  { event := event108872
    frameStart := 108829 },
  { event := event108873
    frameStart := 108829 },
  { event := event108874
    frameStart := 108829 },
  { event := event108875
    frameStart := 108829 },
  { event := event108876
    frameStart := 108829 },
  { event := event108877
    frameStart := 108829 },
  { event := event108878
    frameStart := 108829 },
  { event := event108879
    frameStart := 108829 }
]

def eventLeaf6805 : Array AnnotatedEvent := #[
  { event := event108880
    frameStart := 108829 },
  { event := event108881
    frameStart := 108829 },
  { event := event108882
    frameStart := 108829 },
  { event := event108883
    frameStart := 108883 },
  { event := event108884
    frameStart := 108883 },
  { event := event108885
    frameStart := 108883 },
  { event := event108886
    frameStart := 108883 },
  { event := event108887
    frameStart := 108883 },
  { event := event108888
    frameStart := 108883 },
  { event := event108889
    frameStart := 108883 },
  { event := event108890
    frameStart := 108883 },
  { event := event108891
    frameStart := 108883 },
  { event := event108892
    frameStart := 108883 },
  { event := event108893
    frameStart := 108883 },
  { event := event108894
    frameStart := 108883 },
  { event := event108895
    frameStart := 108883 }
]

def eventLeaf6806 : Array AnnotatedEvent := #[
  { event := event108896
    frameStart := 108883 },
  { event := event108897
    frameStart := 108883 },
  { event := event108898
    frameStart := 108883 },
  { event := event108899
    frameStart := 108883 },
  { event := event108900
    frameStart := 108883 },
  { event := event108901
    frameStart := 108883 },
  { event := event108902
    frameStart := 108883 },
  { event := event108903
    frameStart := 108883 },
  { event := event108904
    frameStart := 108883 },
  { event := event108905
    frameStart := 108883 },
  { event := event108906
    frameStart := 108883 },
  { event := event108907
    frameStart := 108883 },
  { event := event108908
    frameStart := 108883 },
  { event := event108909
    frameStart := 108883 },
  { event := event108910
    frameStart := 108883 },
  { event := event108911
    frameStart := 108883 }
]

def eventLeaf6807 : Array AnnotatedEvent := #[
  { event := event108912
    frameStart := 108883 },
  { event := event108913
    frameStart := 108883 },
  { event := event108914
    frameStart := 108883 },
  { event := event108915
    frameStart := 108883 },
  { event := event108916
    frameStart := 108883 },
  { event := event108917
    frameStart := 108883 },
  { event := event108918
    frameStart := 108883 },
  { event := event108919
    frameStart := 108883 },
  { event := event108920
    frameStart := 108883 },
  { event := event108921
    frameStart := 108883 },
  { event := event108922
    frameStart := 108883 },
  { event := event108923
    frameStart := 108883 },
  { event := event108924
    frameStart := 108883 },
  { event := event108925
    frameStart := 108883 },
  { event := event108926
    frameStart := 108883 },
  { event := event108927
    frameStart := 108883 }
]

def eventLeaf6808 : Array AnnotatedEvent := #[
  { event := event108928
    frameStart := 108883 },
  { event := event108929
    frameStart := 108883 },
  { event := event108930
    frameStart := 108883 },
  { event := event108931
    frameStart := 108883 },
  { event := event108932
    frameStart := 108883 },
  { event := event108933
    frameStart := 108883 },
  { event := event108934
    frameStart := 108883 },
  { event := event108935
    frameStart := 108883 },
  { event := event108936
    frameStart := 108883 },
  { event := event108937
    frameStart := 108883 },
  { event := event108938
    frameStart := 108883 },
  { event := event108939
    frameStart := 108883 },
  { event := event108940
    frameStart := 108883 },
  { event := event108941
    frameStart := 108883 },
  { event := event108942
    frameStart := 108883 },
  { event := event108943
    frameStart := 108883 }
]

def eventLeaf6809 : Array AnnotatedEvent := #[
  { event := event108944
    frameStart := 108883 },
  { event := event108945
    frameStart := 108883 },
  { event := event108946
    frameStart := 108883 },
  { event := event108947
    frameStart := 108883 },
  { event := event108948
    frameStart := 108883 },
  { event := event108949
    frameStart := 108883 },
  { event := event108950
    frameStart := 108883 },
  { event := event108951
    frameStart := 108883 },
  { event := event108952
    frameStart := 108883 },
  { event := event108953
    frameStart := 108883 },
  { event := event108954
    frameStart := 108883 },
  { event := event108955
    frameStart := 108883 },
  { event := event108956
    frameStart := 108883 },
  { event := event108957
    frameStart := 108883 },
  { event := event108958
    frameStart := 108883 },
  { event := event108959
    frameStart := 108883 }
]

def eventLeaf6810 : Array AnnotatedEvent := #[
  { event := event108960
    frameStart := 108883 },
  { event := event108961
    frameStart := 108883 },
  { event := event108962
    frameStart := 108883 },
  { event := event108963
    frameStart := 108883 },
  { event := event108964
    frameStart := 108883 },
  { event := event108965
    frameStart := 108883 },
  { event := event108966
    frameStart := 108883 },
  { event := event108967
    frameStart := 108883 },
  { event := event108968
    frameStart := 108883 },
  { event := event108969
    frameStart := 108883 },
  { event := event108970
    frameStart := 108883 },
  { event := event108971
    frameStart := 108883 },
  { event := event108972
    frameStart := 108883 },
  { event := event108973
    frameStart := 108883 },
  { event := event108974
    frameStart := 108883 },
  { event := event108975
    frameStart := 108883 }
]

def eventLeaf6811 : Array AnnotatedEvent := #[
  { event := event108976
    frameStart := 108883 },
  { event := event108977
    frameStart := 108883 },
  { event := event108978
    frameStart := 108883 },
  { event := event108979
    frameStart := 108883 },
  { event := event108980
    frameStart := 108883 },
  { event := event108981
    frameStart := 108883 },
  { event := event108982
    frameStart := 108883 },
  { event := event108983
    frameStart := 108883 },
  { event := event108984
    frameStart := 108883 },
  { event := event108985
    frameStart := 108883 },
  { event := event108986
    frameStart := 108883 },
  { event := event108987
    frameStart := 0 },
  { event := event108988
    frameStart := 0 },
  { event := event108989
    frameStart := 0 },
  { event := event108990
    frameStart := 0 },
  { event := event108991
    frameStart := 0 }
]

def eventLeaf6812 : Array AnnotatedEvent := #[
  { event := event108992
    frameStart := 0 },
  { event := event108993
    frameStart := 0 },
  { event := event108994
    frameStart := 0 },
  { event := event108995
    frameStart := 0 },
  { event := event108996
    frameStart := 0 },
  { event := event108997
    frameStart := 0 },
  { event := event108998
    frameStart := 0 },
  { event := event108999
    frameStart := 0 },
  { event := event109000
    frameStart := 0 },
  { event := event109001
    frameStart := 0 },
  { event := event109002
    frameStart := 0 },
  { event := event109003
    frameStart := 0 },
  { event := event109004
    frameStart := 0 },
  { event := event109005
    frameStart := 0 },
  { event := event109006
    frameStart := 0 },
  { event := event109007
    frameStart := 0 }
]

def eventLeaf6813 : Array AnnotatedEvent := #[
  { event := event109008
    frameStart := 0 },
  { event := event109009
    frameStart := 0 },
  { event := event109010
    frameStart := 0 },
  { event := event109011
    frameStart := 0 },
  { event := event109012
    frameStart := 0 },
  { event := event109013
    frameStart := 0 },
  { event := event109014
    frameStart := 0 },
  { event := event109015
    frameStart := 0 },
  { event := event109016
    frameStart := 0 },
  { event := event109017
    frameStart := 0 },
  { event := event109018
    frameStart := 0 },
  { event := event109019
    frameStart := 0 },
  { event := event109020
    frameStart := 0 },
  { event := event109021
    frameStart := 0 },
  { event := event109022
    frameStart := 0 },
  { event := event109023
    frameStart := 0 }
]

def eventLeaf6814 : Array AnnotatedEvent := #[
  { event := event109024
    frameStart := 0 },
  { event := event109025
    frameStart := 0 },
  { event := event109026
    frameStart := 0 },
  { event := event109027
    frameStart := 0 },
  { event := event109028
    frameStart := 0 },
  { event := event109029
    frameStart := 0 },
  { event := event109030
    frameStart := 0 },
  { event := event109031
    frameStart := 0 },
  { event := event109032
    frameStart := 0 },
  { event := event109033
    frameStart := 0 },
  { event := event109034
    frameStart := 0 },
  { event := event109035
    frameStart := 0 },
  { event := event109036
    frameStart := 0 },
  { event := event109037
    frameStart := 0 },
  { event := event109038
    frameStart := 0 },
  { event := event109039
    frameStart := 0 }
]

def eventLeaf6815 : Array AnnotatedEvent := #[
  { event := event109040
    frameStart := 0 },
  { event := event109041
    frameStart := 0 },
  { event := event109042
    frameStart := 0 },
  { event := event109043
    frameStart := 0 },
  { event := event109044
    frameStart := 0 },
  { event := event109045
    frameStart := 0 },
  { event := event109046
    frameStart := 0 },
  { event := event109047
    frameStart := 0 },
  { event := event109048
    frameStart := 0 },
  { event := event109049
    frameStart := 0 },
  { event := event109050
    frameStart := 0 },
  { event := event109051
    frameStart := 0 },
  { event := event109052
    frameStart := 0 },
  { event := event109053
    frameStart := 0 },
  { event := event109054
    frameStart := 0 },
  { event := event109055
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events425
