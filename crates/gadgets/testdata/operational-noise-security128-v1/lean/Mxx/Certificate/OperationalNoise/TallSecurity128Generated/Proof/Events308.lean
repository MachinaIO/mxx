import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events308

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact78848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78848RawTermsValid :
    exact78848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13373⟩⟩) exact78848RawTerms .large 78847 .exactZero (none)

def event78849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13374⟩⟩) 0 ⟨13373⟩ 78848

def event78850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13374⟩⟩) 1 ⟨122⟩ 20119

def event78851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13374⟩⟩) (.sum [.predecessor 0 78849 .coefficient, .predecessor 1 78850 .coefficient])

def event78852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13374⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨122⟩⟩]⟩) [⟨.result 20119 .coefficient, false, none⟩])

def event78853 : Event := .survivorFold (1) 78852

def exact78854RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78854RawTermsValid :
    exact78854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13374⟩⟩) exact78854RawTerms .large 78851 (.finite 26) (some (78852))

def event78855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13375⟩⟩) 0 ⟨13374⟩ 78854

def event78856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13375⟩⟩) 1 ⟨9548⟩ 20116

def event78857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13375⟩⟩) (.product (.predecessor 0 78855 .coefficient) (.predecessor 1 78856 .coefficient) (⟨false, false, none, none, none⟩))

def event78858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13375⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) [⟨.result 20112 .coefficient, false, none⟩])

def event78859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13375⟩⟩) (.product (.result 78854 .summary) (.transfer 78858) (⟨false, false, none, none, none⟩))

def event78860 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13375⟩⟩, .operator (⟨78854, 1⟩, ⟨20116, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (-1)⟩)

def event78861 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13375⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9547⟩⟩) ⟨7279⟩ 20086)

def event78862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13375⟩⟩, .relation 78861 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (-1)⟩)

def event78863 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13375⟩⟩, .operator (⟨78854, 0⟩, ⟨20116, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩)

def exact78864RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (-1)⟩]

theorem exact78864RawTermsValid :
    exact78864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78864 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13375⟩⟩) exact78864RawTerms .large 78857 (.finite 279172874240) (some (78859))

def event78865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28925⟩⟩) 0 ⟨13375⟩ 78864

def event78866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28925⟩⟩) 1 ⟨28924⟩ 78834

def event78867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28925⟩⟩) (.sum [.predecessor 0 78865 .coefficient, .predecessor 1 78866 .coefficient])

def event78868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28925⟩⟩, .operator (⟨78864, 1⟩, ⟨78834, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩)

def event78869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28925⟩⟩) (.sum [.result 78864 .summary, .result 78834 .summary])

def exact78870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78870RawTermsValid :
    exact78870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28925⟩⟩) exact78870RawTerms .large 78867 (.finite 279203545088) (some (78869))

def event78871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30666⟩⟩) 0 ⟨28925⟩ 78870

def event78872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30666⟩⟩) 1 ⟨30665⟩ 78806

def event78873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30666⟩⟩) (.product (.predecessor 0 78871 .coefficient) (.predecessor 1 78872 .coefficient) (⟨false, false, none, none, none⟩))

def event78874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30666⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩) [⟨.result 78806 .coefficient, false, none⟩])

def event78875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30666⟩⟩) (.product (.result 78870 .summary) (.transfer 78874) (⟨false, false, none, none, none⟩))

def event78876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30666⟩⟩, .operator (⟨78870, 1⟩, ⟨78806, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩)

def event78877 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30666⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30665⟩⟩) ⟨30125⟩ 78803)

def event78878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30666⟩⟩, .relation 78877 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (-1)⟩)

def event78879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30666⟩⟩, .operator (⟨78870, 0⟩, ⟨78806, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩)

def exact78880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (-1)⟩]

theorem exact78880RawTermsValid :
    exact78880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30666⟩⟩) exact78880RawTerms .large 78873 (.finite 2997925237700553605120) (some (78875))

def event78881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29589⟩⟩) 0 ⟨28920⟩ 3235

def event78882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29589⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact78883RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩]

theorem exact78883RawTermsValid :
    exact78883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78883 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29589⟩⟩) exact78883RawTerms (.finite 5647228698) 78882 .exactZero (none)

def event78884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29591⟩⟩) 0 ⟨29589⟩ 78883

def event78885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29591⟩⟩) 1 ⟨2370⟩ 4

def event78886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29591⟩⟩) (.scale (.predecessor 0 78884 .coefficient) (.value (.predecessor 1 78885 .coefficient)))

def exact78887RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩]

theorem exact78887RawTermsValid :
    exact78887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78887 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29591⟩⟩) exact78887RawTerms (.finite 5647228698) 78886 .exactZero (none)

def event78888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29592⟩⟩) 0 ⟨10368⟩ 75995

def event78889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29592⟩⟩) 1 ⟨29591⟩ 78887

def event78890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29592⟩⟩) (.product (.predecessor 0 78888 .coefficient) (.predecessor 1 78889 .coefficient) (⟨false, false, none, none, none⟩))

def event78891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29592⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩) [⟨.result 78883 .coefficient, false, none⟩])

def event78892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29592⟩⟩) (.product (.result 75995 .summary) (.transfer 78891) (⟨false, false, none, none, none⟩))

def event78893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29592⟩⟩, .operator (⟨75995, 0⟩, ⟨78887, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩)

def event78894 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29590⟩⟩)

def event78895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78896 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78898 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78900 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78902 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78902

def event78904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78900

def event78905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78903 .coefficient) (.value (.predecessor 1 78904 .coefficient)))

def event78906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78906

def event78908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78898

def event78909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78907 .coefficient, .predecessor 1 78908 .coefficient])

def event78910 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78910

def event78912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78896

def event78913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78912 .coefficient))

def event78914 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 78914

def event78916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact78917RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact78917RawTermsValid :
    exact78917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact78917RawTerms (.finite 36) 78916 .exactZero (none)

def event78918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 78914

def event78919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact78920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact78920RawTermsValid :
    exact78920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact78920RawTerms (.finite 36) 78919 .exactZero (none)

def event78921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 78920

def event78922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 78917

def event78923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 78921 .coefficient) (.predecessor 1 78922 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩) [⟨.result 78920 .coefficient, true, some 1⟩, ⟨.result 78917 .coefficient, true, some 1⟩])

def event78925 : Event := .survivorFold (1) 78924

def exact78926RawTerms : List Term := []

theorem exact78926RawTermsValid :
    exact78926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact78926RawTerms (.finite 1296) 78923 (.finite 1296) (some (78924))

def event78927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 78926

def event78928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 78927 .coefficient))

def event78929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event78930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29589⟩⟩) 0 ⟨28920⟩ 78929

def event78931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29589⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact78932RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩]

theorem exact78932RawTermsValid :
    exact78932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29589⟩⟩) exact78932RawTerms (.finite 5647228698) 78931 .exactZero (none)

def event78933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact78934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact78934RawTermsValid :
    exact78934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact78934RawTerms .large 78933 .exactZero (none)

def event78935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29590⟩⟩) 0 ⟨35⟩ 78934

def event78936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29590⟩⟩) 1 ⟨29589⟩ 78932

def event78937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29590⟩⟩) (.product (.predecessor 0 78935 .coefficient) (.predecessor 1 78936 .coefficient) (⟨false, false, none, none, none⟩))

def event78938 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29590⟩⟩, .operator (⟨78934, 0⟩, ⟨78932, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩)

def exact78939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩]

theorem exact78939RawTermsValid :
    exact78939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29590⟩⟩) exact78939RawTerms .large 78937 .exactZero (none)

def event78940 : Event := .preFoldPolynomial 78939 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩] .exactZero none

def exact78941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩, (1)⟩]

def event78941 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29590⟩⟩) 78940 exact78941RawTerms .large 78937 .exactZero (none)

def event78942 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨30669⟩⟩)

def event78943 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78944 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78946 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78948 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78950 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78950

def event78952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78948

def event78953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78951 .coefficient) (.value (.predecessor 1 78952 .coefficient)))

def event78954 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78954

def event78956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78946

def event78957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78955 .coefficient, .predecessor 1 78956 .coefficient])

def event78958 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78958

def event78960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78944

def event78961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78960 .coefficient))

def event78962 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 78962

def event78964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact78965RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact78965RawTermsValid :
    exact78965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact78965RawTerms (.finite 36) 78964 .exactZero (none)

def event78966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 78962

def event78967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact78968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact78968RawTermsValid :
    exact78968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact78968RawTerms (.finite 36) 78967 .exactZero (none)

def event78969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 78968

def event78970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 78965

def event78971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 78969 .coefficient) (.predecessor 1 78970 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78972 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28919⟩⟩, .operator (⟨78968, 0⟩, ⟨78965, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩)

def exact78973RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact78973RawTermsValid :
    exact78973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact78973RawTerms (.finite 1296) 78971 .exactZero (none)

def event78974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 78973

def event78975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 78974 .coefficient))

def event78976 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event78977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30124⟩⟩) 0 ⟨28920⟩ 78976

def event78978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30124⟩⟩) (.authority (.programFamilyFact))

def event78979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30124⟩⟩) (.finite 3720)

def event78980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event78981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30125⟩⟩) 0 ⟨7177⟩ 78980

def event78982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30125⟩⟩) 1 ⟨30124⟩ 78979

def event78983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30125⟩⟩) (.authority (.operator))

def exact78984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩]

theorem exact78984RawTermsValid :
    exact78984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30125⟩⟩) exact78984RawTerms .large 78983 .exactZero (none)

def event78985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30665⟩⟩) 0 ⟨30125⟩ 78984

def event78986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30665⟩⟩) (.authority (.operator))

def exact78987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩]

theorem exact78987RawTermsValid :
    exact78987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30665⟩⟩) exact78987RawTerms (.finite 8192) 78986 .exactZero (none)

def event78988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event78989 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event78990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30390⟩⟩) 0 ⟨28920⟩ 78976

def event78991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30390⟩⟩) 1 ⟨136⟩ 78989

def event78992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30390⟩⟩) (.sum [.predecessor 0 78990 .coefficient, .predecessor 1 78991 .coefficient])

def event78993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30390⟩⟩) (.finite 1296)

def event78994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30391⟩⟩) 0 ⟨30390⟩ 78993

def event78995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30391⟩⟩) (.identity (.predecessor 0 78994 .coefficient))

def exact78996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact78996RawTermsValid :
    exact78996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30391⟩⟩) exact78996RawTerms (.finite 1296) 78995 .exactZero (none)

def event78997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact78998RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78998RawTermsValid :
    exact78998RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78998 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact78998RawTerms .large 78997 .exactZero (none)

def event78999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30392⟩⟩) 0 ⟨6908⟩ 78998

def event79000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30392⟩⟩) 1 ⟨30391⟩ 78996

def event79001 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30392⟩⟩) (.product (.predecessor 0 78999 .coefficient) (.predecessor 1 79000 .coefficient) (⟨false, false, none, none, none⟩))

def event79002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30392⟩⟩, .operator (⟨78998, 0⟩, ⟨78996, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79003RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79003RawTermsValid :
    exact79003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79003 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30392⟩⟩) exact79003RawTerms .large 79001 .exactZero (none)

def event79004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event79005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event79006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 78980

def event79007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact79008RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact79008RawTermsValid :
    exact79008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79008 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact79008RawTerms .large 79007 .exactZero (none)

def event79009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7279⟩⟩) 0 ⟨7178⟩ 79008

def event79010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7279⟩⟩) (.identity (.predecessor 0 79009 .coefficient))

def exact79011RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩]

theorem exact79011RawTermsValid :
    exact79011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79011 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7279⟩⟩) exact79011RawTerms .large 79010 .exactZero (none)

def event79012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9547⟩⟩) 0 ⟨7279⟩ 79011

def event79013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9547⟩⟩) (.authority (.operator))

def exact79014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact79014RawTermsValid :
    exact79014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79014 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9547⟩⟩) exact79014RawTerms (.finite 8192) 79013 .exactZero (none)

def event79015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9548⟩⟩) 0 ⟨9547⟩ 79014

def event79016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9548⟩⟩) 1 ⟨2370⟩ 79005

def event79017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9548⟩⟩) (.scale (.predecessor 0 79015 .coefficient) (.value (.predecessor 1 79016 .coefficient)))

def exact79018RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact79018RawTermsValid :
    exact79018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9548⟩⟩) exact79018RawTerms (.finite 8192) 79017 .exactZero (none)

def event79019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7296⟩⟩) 0 ⟨7178⟩ 79008

def event79020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7296⟩⟩) (.identity (.predecessor 0 79019 .coefficient))

def exact79021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩]

theorem exact79021RawTermsValid :
    exact79021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7296⟩⟩) exact79021RawTerms .large 79020 .exactZero (none)

def event79022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9549⟩⟩) 0 ⟨7296⟩ 79021

def event79023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9549⟩⟩) 1 ⟨9548⟩ 79018

def event79024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9549⟩⟩) (.product (.predecessor 0 79022 .coefficient) (.predecessor 1 79023 .coefficient) (⟨false, false, none, none, none⟩))

def event79025 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9549⟩⟩, .operator (⟨79021, 0⟩, ⟨79018, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩)

def exact79026RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact79026RawTermsValid :
    exact79026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79026 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9549⟩⟩) exact79026RawTerms .large 79024 .exactZero (none)

def event79027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30393⟩⟩) 0 ⟨9549⟩ 79026

def event79028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30393⟩⟩) 1 ⟨30392⟩ 79003

def event79029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30393⟩⟩) (.sum [.predecessor 0 79027 .coefficient, .predecessor 1 79028 .coefficient])

def exact79030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79030RawTermsValid :
    exact79030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30393⟩⟩) exact79030RawTerms .large 79029 .exactZero (none)

def event79031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30668⟩⟩) 0 ⟨30393⟩ 79030

def event79032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30668⟩⟩) 1 ⟨30665⟩ 78987

def event79033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30668⟩⟩) (.product (.predecessor 0 79031 .coefficient) (.predecessor 1 79032 .coefficient) (⟨false, false, none, none, none⟩))

def event79034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30668⟩⟩, .operator (⟨79030, 0⟩, ⟨78987, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩)

def event79035 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30668⟩⟩, .operator (⟨79030, 1⟩, ⟨78987, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩)

def event79036 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30668⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30665⟩⟩) ⟨30125⟩ 78984)

def event79037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30668⟩⟩, .relation 79036 0, ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (-1)⟩)

def exact79038RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (-1)⟩]

theorem exact79038RawTermsValid :
    exact79038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30668⟩⟩) exact79038RawTerms .large 79033 .exactZero (none)

def event79039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29136⟩⟩) 0 ⟨28920⟩ 78976

def event79040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29136⟩⟩) (.authority (.programFamilyFact))

def exact79041RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact79041RawTermsValid :
    exact79041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79041 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29136⟩⟩) exact79041RawTerms (.finite 36) 79040 .exactZero (none)

def event79042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29138⟩⟩) 0 ⟨6908⟩ 78998

def event79043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29138⟩⟩) 1 ⟨29136⟩ 79041

def event79044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29138⟩⟩) (.product (.predecessor 0 79042 .coefficient) (.predecessor 1 79043 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79045 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29138⟩⟩, .operator (⟨78998, 0⟩, ⟨79041, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79046RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79046RawTermsValid :
    exact79046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79046 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29138⟩⟩) exact79046RawTerms .large 79044 .exactZero (none)

def event79047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 78980

def event79048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact79049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact79049RawTermsValid :
    exact79049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact79049RawTerms .large 79048 .exactZero (none)

def event79050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29139⟩⟩) 0 ⟨7190⟩ 79049

def event79051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29139⟩⟩) 1 ⟨29138⟩ 79046

def event79052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29139⟩⟩) (.sum [.predecessor 0 79050 .coefficient, .predecessor 1 79051 .coefficient])

def exact79053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79053RawTermsValid :
    exact79053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29139⟩⟩) exact79053RawTerms .large 79052 .exactZero (none)

def event79054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30669⟩⟩) 0 ⟨29139⟩ 79053

def event79055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30669⟩⟩) 1 ⟨30668⟩ 79038

def event79056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30669⟩⟩) (.sum [.predecessor 0 79054 .coefficient, .predecessor 1 79055 .coefficient])

def exact79057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79057RawTermsValid :
    exact79057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79057 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30669⟩⟩) exact79057RawTerms .large 79056 .exactZero (none)

def event79058 : Event := .preFoldPolynomial 79057 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact79059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event79059 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨30669⟩⟩) 79058 exact79059RawTerms .large 79056 .exactZero (none)

def event79060 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨28920⟩⟩) ⟨⟨69⟩, ⟨48⟩, ⟨135⟩⟩ ⟨78894, 79060⟩

def event79061 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29592⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩) (1) 0 2 (.universal 79060 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29589⟩⟩]⟩) (none) 79059)

def event79062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29592⟩⟩, .relation 79061 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩)

def event79063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29592⟩⟩, .relation 79061 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩)

def event79064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29592⟩⟩, .relation 79061 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩)

def event79065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29592⟩⟩, .relation 79061 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact79066RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79066RawTermsValid :
    exact79066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29592⟩⟩) exact79066RawTerms .large 78890 (.finite 202072841853861888) (some (78892))

def event79067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30667⟩⟩) 0 ⟨29592⟩ 79066

def event79068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30667⟩⟩) 1 ⟨30666⟩ 78880

def event79069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30667⟩⟩) (.sum [.predecessor 0 79067 .coefficient, .predecessor 1 79068 .coefficient])

def event79070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30667⟩⟩, .operator (⟨79066, 2⟩, ⟨78880, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (-1)⟩)

def event79071 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30667⟩⟩, .operator (⟨79066, 1⟩, ⟨78880, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩)

def event79072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30667⟩⟩) (.sum [.result 79066 .summary, .result 78880 .summary])

def exact79073RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79073RawTermsValid :
    exact79073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30667⟩⟩) exact79073RawTerms .large 79069 (.finite 2998127310542407467008) (some (79072))

def event79074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31121⟩⟩) 0 ⟨30667⟩ 79073

def event79075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31121⟩⟩) 1 ⟨31119⟩ 78796

def event79076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31121⟩⟩) (.product (.predecessor 0 79074 .coefficient) (.predecessor 1 79075 .coefficient) (⟨false, false, none, none, none⟩))

def event79077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31121⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩) [⟨.result 78796 .coefficient, false, none⟩])

def event79078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31121⟩⟩) (.product (.result 79073 .summary) (.transfer 79077) (⟨false, false, none, none, none⟩))

def event79079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31121⟩⟩, .operator (⟨79073, 0⟩, ⟨78796, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩)

def event79080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31121⟩⟩, .operator (⟨79073, 1⟩, ⟨78796, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩)

def event79081 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31121⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31119⟩⟩) ⟨30295⟩ 78793)

def event79082 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31121⟩⟩, .relation 79081 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (-1)⟩)

def exact79083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (-1)⟩]

theorem exact79083RawTermsValid :
    exact79083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31121⟩⟩) exact79083RawTerms .large 79076 (.finite 32192146870060190229763897425920) (some (79078))

def event79084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29956⟩⟩) 0 ⟨29137⟩ 3241

def event79085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29956⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact79086RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩]

theorem exact79086RawTermsValid :
    exact79086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79086 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29956⟩⟩) exact79086RawTerms (.finite 5647228698) 79085 .exactZero (none)

def event79087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29958⟩⟩) 0 ⟨29956⟩ 79086

def event79088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29958⟩⟩) 1 ⟨2370⟩ 4

def event79089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29958⟩⟩) (.scale (.predecessor 0 79087 .coefficient) (.value (.predecessor 1 79088 .coefficient)))

def exact79090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩]

theorem exact79090RawTermsValid :
    exact79090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29958⟩⟩) exact79090RawTerms (.finite 5647228698) 79089 .exactZero (none)

def event79091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29959⟩⟩) 0 ⟨10368⟩ 75995

def event79092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29959⟩⟩) 1 ⟨29958⟩ 79090

def event79093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29959⟩⟩) (.product (.predecessor 0 79091 .coefficient) (.predecessor 1 79092 .coefficient) (⟨false, false, none, none, none⟩))

def event79094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29959⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩) [⟨.result 79086 .coefficient, false, none⟩])

def event79095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29959⟩⟩) (.product (.result 75995 .summary) (.transfer 79094) (⟨false, false, none, none, none⟩))

def event79096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29959⟩⟩, .operator (⟨75995, 0⟩, ⟨79090, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩)

def event79097 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29957⟩⟩)

def event79098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79099 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79103 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def eventLeaf4928 : Array AnnotatedEvent := #[
  { event := event78848
    frameStart := 0 },
  { event := event78849
    frameStart := 0 },
  { event := event78850
    frameStart := 0 },
  { event := event78851
    frameStart := 0 },
  { event := event78852
    frameStart := 0 },
  { event := event78853
    frameStart := 0 },
  { event := event78854
    frameStart := 0 },
  { event := event78855
    frameStart := 0 },
  { event := event78856
    frameStart := 0 },
  { event := event78857
    frameStart := 0 },
  { event := event78858
    frameStart := 0 },
  { event := event78859
    frameStart := 0 },
  { event := event78860
    frameStart := 0 },
  { event := event78861
    frameStart := 0 },
  { event := event78862
    frameStart := 0 },
  { event := event78863
    frameStart := 0 }
]

def eventLeaf4929 : Array AnnotatedEvent := #[
  { event := event78864
    frameStart := 0 },
  { event := event78865
    frameStart := 0 },
  { event := event78866
    frameStart := 0 },
  { event := event78867
    frameStart := 0 },
  { event := event78868
    frameStart := 0 },
  { event := event78869
    frameStart := 0 },
  { event := event78870
    frameStart := 0 },
  { event := event78871
    frameStart := 0 },
  { event := event78872
    frameStart := 0 },
  { event := event78873
    frameStart := 0 },
  { event := event78874
    frameStart := 0 },
  { event := event78875
    frameStart := 0 },
  { event := event78876
    frameStart := 0 },
  { event := event78877
    frameStart := 0 },
  { event := event78878
    frameStart := 0 },
  { event := event78879
    frameStart := 0 }
]

def eventLeaf4930 : Array AnnotatedEvent := #[
  { event := event78880
    frameStart := 0 },
  { event := event78881
    frameStart := 0 },
  { event := event78882
    frameStart := 0 },
  { event := event78883
    frameStart := 0 },
  { event := event78884
    frameStart := 0 },
  { event := event78885
    frameStart := 0 },
  { event := event78886
    frameStart := 0 },
  { event := event78887
    frameStart := 0 },
  { event := event78888
    frameStart := 0 },
  { event := event78889
    frameStart := 0 },
  { event := event78890
    frameStart := 0 },
  { event := event78891
    frameStart := 0 },
  { event := event78892
    frameStart := 0 },
  { event := event78893
    frameStart := 0 },
  { event := event78894
    frameStart := 78894 },
  { event := event78895
    frameStart := 78894 }
]

def eventLeaf4931 : Array AnnotatedEvent := #[
  { event := event78896
    frameStart := 78894 },
  { event := event78897
    frameStart := 78894 },
  { event := event78898
    frameStart := 78894 },
  { event := event78899
    frameStart := 78894 },
  { event := event78900
    frameStart := 78894 },
  { event := event78901
    frameStart := 78894 },
  { event := event78902
    frameStart := 78894 },
  { event := event78903
    frameStart := 78894 },
  { event := event78904
    frameStart := 78894 },
  { event := event78905
    frameStart := 78894 },
  { event := event78906
    frameStart := 78894 },
  { event := event78907
    frameStart := 78894 },
  { event := event78908
    frameStart := 78894 },
  { event := event78909
    frameStart := 78894 },
  { event := event78910
    frameStart := 78894 },
  { event := event78911
    frameStart := 78894 }
]

def eventLeaf4932 : Array AnnotatedEvent := #[
  { event := event78912
    frameStart := 78894 },
  { event := event78913
    frameStart := 78894 },
  { event := event78914
    frameStart := 78894 },
  { event := event78915
    frameStart := 78894 },
  { event := event78916
    frameStart := 78894 },
  { event := event78917
    frameStart := 78894 },
  { event := event78918
    frameStart := 78894 },
  { event := event78919
    frameStart := 78894 },
  { event := event78920
    frameStart := 78894 },
  { event := event78921
    frameStart := 78894 },
  { event := event78922
    frameStart := 78894 },
  { event := event78923
    frameStart := 78894 },
  { event := event78924
    frameStart := 78894 },
  { event := event78925
    frameStart := 78894 },
  { event := event78926
    frameStart := 78894 },
  { event := event78927
    frameStart := 78894 }
]

def eventLeaf4933 : Array AnnotatedEvent := #[
  { event := event78928
    frameStart := 78894 },
  { event := event78929
    frameStart := 78894 },
  { event := event78930
    frameStart := 78894 },
  { event := event78931
    frameStart := 78894 },
  { event := event78932
    frameStart := 78894 },
  { event := event78933
    frameStart := 78894 },
  { event := event78934
    frameStart := 78894 },
  { event := event78935
    frameStart := 78894 },
  { event := event78936
    frameStart := 78894 },
  { event := event78937
    frameStart := 78894 },
  { event := event78938
    frameStart := 78894 },
  { event := event78939
    frameStart := 78894 },
  { event := event78940
    frameStart := 78894 },
  { event := event78941
    frameStart := 78894 },
  { event := event78942
    frameStart := 78942 },
  { event := event78943
    frameStart := 78942 }
]

def eventLeaf4934 : Array AnnotatedEvent := #[
  { event := event78944
    frameStart := 78942 },
  { event := event78945
    frameStart := 78942 },
  { event := event78946
    frameStart := 78942 },
  { event := event78947
    frameStart := 78942 },
  { event := event78948
    frameStart := 78942 },
  { event := event78949
    frameStart := 78942 },
  { event := event78950
    frameStart := 78942 },
  { event := event78951
    frameStart := 78942 },
  { event := event78952
    frameStart := 78942 },
  { event := event78953
    frameStart := 78942 },
  { event := event78954
    frameStart := 78942 },
  { event := event78955
    frameStart := 78942 },
  { event := event78956
    frameStart := 78942 },
  { event := event78957
    frameStart := 78942 },
  { event := event78958
    frameStart := 78942 },
  { event := event78959
    frameStart := 78942 }
]

def eventLeaf4935 : Array AnnotatedEvent := #[
  { event := event78960
    frameStart := 78942 },
  { event := event78961
    frameStart := 78942 },
  { event := event78962
    frameStart := 78942 },
  { event := event78963
    frameStart := 78942 },
  { event := event78964
    frameStart := 78942 },
  { event := event78965
    frameStart := 78942 },
  { event := event78966
    frameStart := 78942 },
  { event := event78967
    frameStart := 78942 },
  { event := event78968
    frameStart := 78942 },
  { event := event78969
    frameStart := 78942 },
  { event := event78970
    frameStart := 78942 },
  { event := event78971
    frameStart := 78942 },
  { event := event78972
    frameStart := 78942 },
  { event := event78973
    frameStart := 78942 },
  { event := event78974
    frameStart := 78942 },
  { event := event78975
    frameStart := 78942 }
]

def eventLeaf4936 : Array AnnotatedEvent := #[
  { event := event78976
    frameStart := 78942 },
  { event := event78977
    frameStart := 78942 },
  { event := event78978
    frameStart := 78942 },
  { event := event78979
    frameStart := 78942 },
  { event := event78980
    frameStart := 78942 },
  { event := event78981
    frameStart := 78942 },
  { event := event78982
    frameStart := 78942 },
  { event := event78983
    frameStart := 78942 },
  { event := event78984
    frameStart := 78942 },
  { event := event78985
    frameStart := 78942 },
  { event := event78986
    frameStart := 78942 },
  { event := event78987
    frameStart := 78942 },
  { event := event78988
    frameStart := 78942 },
  { event := event78989
    frameStart := 78942 },
  { event := event78990
    frameStart := 78942 },
  { event := event78991
    frameStart := 78942 }
]

def eventLeaf4937 : Array AnnotatedEvent := #[
  { event := event78992
    frameStart := 78942 },
  { event := event78993
    frameStart := 78942 },
  { event := event78994
    frameStart := 78942 },
  { event := event78995
    frameStart := 78942 },
  { event := event78996
    frameStart := 78942 },
  { event := event78997
    frameStart := 78942 },
  { event := event78998
    frameStart := 78942 },
  { event := event78999
    frameStart := 78942 },
  { event := event79000
    frameStart := 78942 },
  { event := event79001
    frameStart := 78942 },
  { event := event79002
    frameStart := 78942 },
  { event := event79003
    frameStart := 78942 },
  { event := event79004
    frameStart := 78942 },
  { event := event79005
    frameStart := 78942 },
  { event := event79006
    frameStart := 78942 },
  { event := event79007
    frameStart := 78942 }
]

def eventLeaf4938 : Array AnnotatedEvent := #[
  { event := event79008
    frameStart := 78942 },
  { event := event79009
    frameStart := 78942 },
  { event := event79010
    frameStart := 78942 },
  { event := event79011
    frameStart := 78942 },
  { event := event79012
    frameStart := 78942 },
  { event := event79013
    frameStart := 78942 },
  { event := event79014
    frameStart := 78942 },
  { event := event79015
    frameStart := 78942 },
  { event := event79016
    frameStart := 78942 },
  { event := event79017
    frameStart := 78942 },
  { event := event79018
    frameStart := 78942 },
  { event := event79019
    frameStart := 78942 },
  { event := event79020
    frameStart := 78942 },
  { event := event79021
    frameStart := 78942 },
  { event := event79022
    frameStart := 78942 },
  { event := event79023
    frameStart := 78942 }
]

def eventLeaf4939 : Array AnnotatedEvent := #[
  { event := event79024
    frameStart := 78942 },
  { event := event79025
    frameStart := 78942 },
  { event := event79026
    frameStart := 78942 },
  { event := event79027
    frameStart := 78942 },
  { event := event79028
    frameStart := 78942 },
  { event := event79029
    frameStart := 78942 },
  { event := event79030
    frameStart := 78942 },
  { event := event79031
    frameStart := 78942 },
  { event := event79032
    frameStart := 78942 },
  { event := event79033
    frameStart := 78942 },
  { event := event79034
    frameStart := 78942 },
  { event := event79035
    frameStart := 78942 },
  { event := event79036
    frameStart := 78942 },
  { event := event79037
    frameStart := 78942 },
  { event := event79038
    frameStart := 78942 },
  { event := event79039
    frameStart := 78942 }
]

def eventLeaf4940 : Array AnnotatedEvent := #[
  { event := event79040
    frameStart := 78942 },
  { event := event79041
    frameStart := 78942 },
  { event := event79042
    frameStart := 78942 },
  { event := event79043
    frameStart := 78942 },
  { event := event79044
    frameStart := 78942 },
  { event := event79045
    frameStart := 78942 },
  { event := event79046
    frameStart := 78942 },
  { event := event79047
    frameStart := 78942 },
  { event := event79048
    frameStart := 78942 },
  { event := event79049
    frameStart := 78942 },
  { event := event79050
    frameStart := 78942 },
  { event := event79051
    frameStart := 78942 },
  { event := event79052
    frameStart := 78942 },
  { event := event79053
    frameStart := 78942 },
  { event := event79054
    frameStart := 78942 },
  { event := event79055
    frameStart := 78942 }
]

def eventLeaf4941 : Array AnnotatedEvent := #[
  { event := event79056
    frameStart := 78942 },
  { event := event79057
    frameStart := 78942 },
  { event := event79058
    frameStart := 78942 },
  { event := event79059
    frameStart := 78942 },
  { event := event79060
    frameStart := 0 },
  { event := event79061
    frameStart := 0 },
  { event := event79062
    frameStart := 0 },
  { event := event79063
    frameStart := 0 },
  { event := event79064
    frameStart := 0 },
  { event := event79065
    frameStart := 0 },
  { event := event79066
    frameStart := 0 },
  { event := event79067
    frameStart := 0 },
  { event := event79068
    frameStart := 0 },
  { event := event79069
    frameStart := 0 },
  { event := event79070
    frameStart := 0 },
  { event := event79071
    frameStart := 0 }
]

def eventLeaf4942 : Array AnnotatedEvent := #[
  { event := event79072
    frameStart := 0 },
  { event := event79073
    frameStart := 0 },
  { event := event79074
    frameStart := 0 },
  { event := event79075
    frameStart := 0 },
  { event := event79076
    frameStart := 0 },
  { event := event79077
    frameStart := 0 },
  { event := event79078
    frameStart := 0 },
  { event := event79079
    frameStart := 0 },
  { event := event79080
    frameStart := 0 },
  { event := event79081
    frameStart := 0 },
  { event := event79082
    frameStart := 0 },
  { event := event79083
    frameStart := 0 },
  { event := event79084
    frameStart := 0 },
  { event := event79085
    frameStart := 0 },
  { event := event79086
    frameStart := 0 },
  { event := event79087
    frameStart := 0 }
]

def eventLeaf4943 : Array AnnotatedEvent := #[
  { event := event79088
    frameStart := 0 },
  { event := event79089
    frameStart := 0 },
  { event := event79090
    frameStart := 0 },
  { event := event79091
    frameStart := 0 },
  { event := event79092
    frameStart := 0 },
  { event := event79093
    frameStart := 0 },
  { event := event79094
    frameStart := 0 },
  { event := event79095
    frameStart := 0 },
  { event := event79096
    frameStart := 0 },
  { event := event79097
    frameStart := 79097 },
  { event := event79098
    frameStart := 79097 },
  { event := event79099
    frameStart := 79097 },
  { event := event79100
    frameStart := 79097 },
  { event := event79101
    frameStart := 79097 },
  { event := event79102
    frameStart := 79097 },
  { event := event79103
    frameStart := 79097 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events308
