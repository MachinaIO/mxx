import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events379

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event97024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9582⟩⟩, .operator (⟨97020, 0⟩, ⟨97017, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact97025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact97025RawTermsValid :
    exact97025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9582⟩⟩) exact97025RawTerms .large 97023 .exactZero (none)

def event97026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52309⟩⟩) 0 ⟨9582⟩ 97025

def event97027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52309⟩⟩) 1 ⟨52308⟩ 97002

def event97028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52309⟩⟩) (.sum [.predecessor 0 97026 .coefficient, .predecessor 1 97027 .coefficient])

def exact97029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97029RawTermsValid :
    exact97029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52309⟩⟩) exact97029RawTerms .large 97028 .exactZero (none)

def event97030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52577⟩⟩) 0 ⟨52309⟩ 97029

def event97031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52577⟩⟩) 1 ⟨52574⟩ 96986

def event97032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52577⟩⟩) (.product (.predecessor 0 97030 .coefficient) (.predecessor 1 97031 .coefficient) (⟨false, false, none, none, none⟩))

def event97033 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52577⟩⟩, .operator (⟨97029, 0⟩, ⟨96986, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩)

def event97034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52577⟩⟩, .operator (⟨97029, 1⟩, ⟨96986, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩)

def event97035 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52577⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52574⟩⟩) ⟨52039⟩ 96983)

def event97036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52577⟩⟩, .relation 97035 0, ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (-1)⟩)

def exact97037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (-1)⟩]

theorem exact97037RawTermsValid :
    exact97037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52577⟩⟩) exact97037RawTerms .large 97032 .exactZero (none)

def event97038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50928⟩⟩) 0 ⟨50682⟩ 96975

def event97039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50928⟩⟩) (.authority (.programFamilyFact))

def exact97040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], []⟩, (1)⟩]

theorem exact97040RawTermsValid :
    exact97040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50928⟩⟩) exact97040RawTerms (.finite 10) 97039 .exactZero (none)

def event97041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50930⟩⟩) 0 ⟨6908⟩ 96997

def event97042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50930⟩⟩) 1 ⟨50928⟩ 97040

def event97043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50930⟩⟩) (.product (.predecessor 0 97041 .coefficient) (.predecessor 1 97042 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50930⟩⟩, .operator (⟨96997, 0⟩, ⟨97040, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97045RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97045RawTermsValid :
    exact97045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97045 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50930⟩⟩) exact97045RawTerms .large 97043 .exactZero (none)

def event97046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 96979

def event97047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact97048RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact97048RawTermsValid :
    exact97048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact97048RawTerms .large 97047 .exactZero (none)

def event97049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50931⟩⟩) 0 ⟨7183⟩ 97048

def event97050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50931⟩⟩) 1 ⟨50930⟩ 97045

def event97051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50931⟩⟩) (.sum [.predecessor 0 97049 .coefficient, .predecessor 1 97050 .coefficient])

def exact97052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97052RawTermsValid :
    exact97052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50931⟩⟩) exact97052RawTerms .large 97051 .exactZero (none)

def event97053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52578⟩⟩) 0 ⟨50931⟩ 97052

def event97054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52578⟩⟩) 1 ⟨52577⟩ 97037

def event97055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52578⟩⟩) (.sum [.predecessor 0 97053 .coefficient, .predecessor 1 97054 .coefficient])

def exact97056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97056RawTermsValid :
    exact97056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52578⟩⟩) exact97056RawTerms .large 97055 .exactZero (none)

def event97057 : Event := .preFoldPolynomial 97056 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact97058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event97058 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52578⟩⟩) 97057 exact97058RawTerms .large 97055 .exactZero (none)

def event97059 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50682⟩⟩) ⟨⟨62⟩, ⟨40⟩, ⟨135⟩⟩ ⟨96893, 97059⟩

def event97060 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51502⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (1) 0 2 (.universal 97059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (none) 97058)

def event97061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51502⟩⟩, .relation 97060 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩)

def event97062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51502⟩⟩, .relation 97060 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩)

def event97063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51502⟩⟩, .relation 97060 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩)

def event97064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51502⟩⟩, .relation 97060 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact97065RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97065RawTermsValid :
    exact97065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51502⟩⟩) exact97065RawTerms .large 96889 (.finite 202072841853861888) (some (96891))

def event97066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52576⟩⟩) 0 ⟨51502⟩ 97065

def event97067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52576⟩⟩) 1 ⟨52575⟩ 96879

def event97068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52576⟩⟩) (.sum [.predecessor 0 97066 .coefficient, .predecessor 1 97067 .coefficient])

def event97069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52576⟩⟩, .operator (⟨97065, 2⟩, ⟨96879, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩, (-1)⟩)

def event97070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52576⟩⟩, .operator (⟨97065, 1⟩, ⟨96879, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩, (1)⟩)

def event97071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52576⟩⟩) (.sum [.result 97065 .summary, .result 96879 .summary])

def exact97072RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97072RawTermsValid :
    exact97072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52576⟩⟩) exact97072RawTerms .large 97068 (.finite 2997889464187086962688) (some (97071))

def event97073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53109⟩⟩) 0 ⟨52576⟩ 97072

def event97074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53109⟩⟩) 1 ⟨53107⟩ 96795

def event97075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53109⟩⟩) (.product (.predecessor 0 97073 .coefficient) (.predecessor 1 97074 .coefficient) (⟨false, false, none, none, none⟩))

def event97076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53109⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩) [⟨.result 96795 .coefficient, false, none⟩])

def event97077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53109⟩⟩) (.product (.result 97072 .summary) (.transfer 97076) (⟨false, false, none, none, none⟩))

def event97078 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53109⟩⟩, .operator (⟨97072, 0⟩, ⟨96795, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩)

def event97079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53109⟩⟩, .operator (⟨97072, 1⟩, ⟨96795, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩)

def event97080 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53109⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53107⟩⟩) ⟨52206⟩ 96792)

def event97081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53109⟩⟩, .relation 97080 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (-1)⟩)

def exact97082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (-1)⟩]

theorem exact97082RawTermsValid :
    exact97082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53109⟩⟩) exact97082RawTerms .large 97075 (.finite 32189593014266254325632330629120) (some (97077))

def event97083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51856⟩⟩) 0 ⟨50929⟩ 4150

def event97084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51856⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact97085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩]

theorem exact97085RawTermsValid :
    exact97085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51856⟩⟩) exact97085RawTerms (.finite 5647228698) 97084 .exactZero (none)

def event97086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51858⟩⟩) 0 ⟨51856⟩ 97085

def event97087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51858⟩⟩) 1 ⟨2370⟩ 4

def event97088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51858⟩⟩) (.scale (.predecessor 0 97086 .coefficient) (.value (.predecessor 1 97087 .coefficient)))

def exact97089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩]

theorem exact97089RawTermsValid :
    exact97089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97089 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51858⟩⟩) exact97089RawTerms (.finite 5647228698) 97088 .exactZero (none)

def event97090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51859⟩⟩) 0 ⟨9944⟩ 90620

def event97091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51859⟩⟩) 1 ⟨51858⟩ 97089

def event97092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51859⟩⟩) (.product (.predecessor 0 97090 .coefficient) (.predecessor 1 97091 .coefficient) (⟨false, false, none, none, none⟩))

def event97093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51859⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩) [⟨.result 97085 .coefficient, false, none⟩])

def event97094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51859⟩⟩) (.product (.result 90620 .summary) (.transfer 97093) (⟨false, false, none, none, none⟩))

def event97095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51859⟩⟩, .operator (⟨90620, 0⟩, ⟨97089, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩)

def event97096 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51857⟩⟩)

def event97097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event97098 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event97099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event97100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event97101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event97102 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event97103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event97104 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event97105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 97104

def event97106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 97102

def event97107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 97105 .coefficient) (.value (.predecessor 1 97106 .coefficient)))

def event97108 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event97109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 97108

def event97110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 97100

def event97111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 97109 .coefficient, .predecessor 1 97110 .coefficient])

def event97112 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event97113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 97112

def event97114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 97098

def event97115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 97114 .coefficient))

def event97116 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event97117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24590⟩⟩) 0 ⟨9901⟩ 97116

def event97118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24590⟩⟩) (.authority (.programFamilyFact))

def exact97119RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩], []⟩, (1)⟩]

theorem exact97119RawTermsValid :
    exact97119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24590⟩⟩) exact97119RawTerms (.finite 10) 97118 .exactZero (none)

def event97120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50680⟩⟩) 0 ⟨9901⟩ 97116

def event97121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50680⟩⟩) (.authority (.programFamilyFact))

def exact97122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact97122RawTermsValid :
    exact97122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50680⟩⟩) exact97122RawTerms (.finite 10) 97121 .exactZero (none)

def event97123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 0 ⟨50680⟩ 97122

def event97124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 1 ⟨24590⟩ 97119

def event97125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.product (.predecessor 0 97123 .coefficient) (.predecessor 1 97124 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩) [⟨.result 97122 .coefficient, true, some 1⟩, ⟨.result 97119 .coefficient, true, some 1⟩])

def event97127 : Event := .survivorFold (1) 97126

def exact97128RawTerms : List Term := []

theorem exact97128RawTermsValid :
    exact97128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50681⟩⟩) exact97128RawTerms (.finite 100) 97125 (.finite 100) (some (97126))

def event97129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50682⟩⟩) 0 ⟨50681⟩ 97128

def event97130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.identity (.predecessor 0 97129 .coefficient))

def event97131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.finite 100)

def event97132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50928⟩⟩) 0 ⟨50682⟩ 97131

def event97133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50928⟩⟩) (.authority (.programFamilyFact))

def exact97134RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], []⟩, (1)⟩]

theorem exact97134RawTermsValid :
    exact97134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50928⟩⟩) exact97134RawTerms (.finite 10) 97133 .exactZero (none)

def event97135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50929⟩⟩) 0 ⟨50928⟩ 97134

def event97136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50929⟩⟩) (.identity (.predecessor 0 97135 .coefficient))

def event97137 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50929⟩⟩) (.finite 10)

def event97138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51856⟩⟩) 0 ⟨50929⟩ 97137

def event97139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51856⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact97140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩]

theorem exact97140RawTermsValid :
    exact97140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51856⟩⟩) exact97140RawTerms (.finite 5647228698) 97139 .exactZero (none)

def event97141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact97142RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact97142RawTermsValid :
    exact97142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97142 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact97142RawTerms .large 97141 .exactZero (none)

def event97143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51857⟩⟩) 0 ⟨35⟩ 97142

def event97144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51857⟩⟩) 1 ⟨51856⟩ 97140

def event97145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51857⟩⟩) (.product (.predecessor 0 97143 .coefficient) (.predecessor 1 97144 .coefficient) (⟨false, false, none, none, none⟩))

def event97146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51857⟩⟩, .operator (⟨97142, 0⟩, ⟨97140, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩)

def exact97147RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩]

theorem exact97147RawTermsValid :
    exact97147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51857⟩⟩) exact97147RawTerms .large 97145 .exactZero (none)

def event97148 : Event := .preFoldPolynomial 97147 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩] .exactZero none

def exact97149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩, (1)⟩]

def event97149 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51857⟩⟩) 97148 exact97149RawTerms .large 97145 .exactZero (none)

def event97150 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨53112⟩⟩)

def event97151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event97152 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event97153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event97154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event97155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event97156 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event97157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event97158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event97159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 97158

def event97160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 97156

def event97161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 97159 .coefficient) (.value (.predecessor 1 97160 .coefficient)))

def event97162 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event97163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 97162

def event97164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 97154

def event97165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 97163 .coefficient, .predecessor 1 97164 .coefficient])

def event97166 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event97167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 97166

def event97168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 97152

def event97169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 97168 .coefficient))

def event97170 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event97171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24590⟩⟩) 0 ⟨9901⟩ 97170

def event97172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24590⟩⟩) (.authority (.programFamilyFact))

def exact97173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩], []⟩, (1)⟩]

theorem exact97173RawTermsValid :
    exact97173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24590⟩⟩) exact97173RawTerms (.finite 10) 97172 .exactZero (none)

def event97174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50680⟩⟩) 0 ⟨9901⟩ 97170

def event97175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50680⟩⟩) (.authority (.programFamilyFact))

def exact97176RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact97176RawTermsValid :
    exact97176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50680⟩⟩) exact97176RawTerms (.finite 10) 97175 .exactZero (none)

def event97177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 0 ⟨50680⟩ 97176

def event97178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50681⟩⟩) 1 ⟨24590⟩ 97173

def event97179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50681⟩⟩) (.product (.predecessor 0 97177 .coefficient) (.predecessor 1 97178 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50681⟩⟩, .operator (⟨97176, 0⟩, ⟨97173, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩)

def exact97181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], []⟩, (1)⟩]

theorem exact97181RawTermsValid :
    exact97181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50681⟩⟩) exact97181RawTerms (.finite 100) 97179 .exactZero (none)

def event97182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50682⟩⟩) 0 ⟨50681⟩ 97181

def event97183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.identity (.predecessor 0 97182 .coefficient))

def event97184 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50682⟩⟩) (.finite 100)

def event97185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50928⟩⟩) 0 ⟨50682⟩ 97184

def event97186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50928⟩⟩) (.authority (.programFamilyFact))

def exact97187RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], []⟩, (1)⟩]

theorem exact97187RawTermsValid :
    exact97187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50928⟩⟩) exact97187RawTerms (.finite 10) 97186 .exactZero (none)

def event97188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50929⟩⟩) 0 ⟨50928⟩ 97187

def event97189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50929⟩⟩) (.identity (.predecessor 0 97188 .coefficient))

def event97190 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50929⟩⟩) (.finite 10)

def event97191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52204⟩⟩) 0 ⟨50929⟩ 97190

def event97192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52204⟩⟩) (.authority (.programFamilyFact))

def event97193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52204⟩⟩) (.finite 3720)

def event97194 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event97195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52206⟩⟩) 0 ⟨7177⟩ 97194

def event97196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52206⟩⟩) 1 ⟨52204⟩ 97193

def event97197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52206⟩⟩) (.authority (.operator))

def exact97198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩]

theorem exact97198RawTermsValid :
    exact97198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52206⟩⟩) exact97198RawTerms .large 97197 .exactZero (none)

def event97199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53107⟩⟩) 0 ⟨52206⟩ 97198

def event97200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53107⟩⟩) (.authority (.operator))

def exact97201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩]

theorem exact97201RawTermsValid :
    exact97201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53107⟩⟩) exact97201RawTerms (.finite 8192) 97200 .exactZero (none)

def event97202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event97203 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event97204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52386⟩⟩) 0 ⟨50929⟩ 97190

def event97205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52386⟩⟩) 1 ⟨136⟩ 97203

def event97206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52386⟩⟩) (.sum [.predecessor 0 97204 .coefficient, .predecessor 1 97205 .coefficient])

def event97207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52386⟩⟩) (.finite 10)

def event97208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52387⟩⟩) 0 ⟨52386⟩ 97207

def event97209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52387⟩⟩) (.identity (.predecessor 0 97208 .coefficient))

def exact97210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], []⟩, (1)⟩]

theorem exact97210RawTermsValid :
    exact97210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52387⟩⟩) exact97210RawTerms (.finite 10) 97209 .exactZero (none)

def event97211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact97212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97212RawTermsValid :
    exact97212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact97212RawTerms .large 97211 .exactZero (none)

def event97213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52388⟩⟩) 0 ⟨6908⟩ 97212

def event97214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52388⟩⟩) 1 ⟨52387⟩ 97210

def event97215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52388⟩⟩) (.product (.predecessor 0 97213 .coefficient) (.predecessor 1 97214 .coefficient) (⟨false, false, none, none, none⟩))

def event97216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52388⟩⟩, .operator (⟨97212, 0⟩, ⟨97210, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97217RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97217RawTermsValid :
    exact97217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52388⟩⟩) exact97217RawTerms .large 97215 .exactZero (none)

def event97218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 97194

def event97219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact97220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact97220RawTermsValid :
    exact97220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact97220RawTerms .large 97219 .exactZero (none)

def event97221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52389⟩⟩) 0 ⟨7183⟩ 97220

def event97222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52389⟩⟩) 1 ⟨52388⟩ 97217

def event97223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52389⟩⟩) (.sum [.predecessor 0 97221 .coefficient, .predecessor 1 97222 .coefficient])

def exact97224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97224RawTermsValid :
    exact97224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52389⟩⟩) exact97224RawTerms .large 97223 .exactZero (none)

def event97225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53108⟩⟩) 0 ⟨52389⟩ 97224

def event97226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53108⟩⟩) 1 ⟨53107⟩ 97201

def event97227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53108⟩⟩) (.product (.predecessor 0 97225 .coefficient) (.predecessor 1 97226 .coefficient) (⟨false, false, none, none, none⟩))

def event97228 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53108⟩⟩, .operator (⟨97224, 0⟩, ⟨97201, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩)

def event97229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53108⟩⟩, .operator (⟨97224, 1⟩, ⟨97201, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩)

def event97230 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53108⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53107⟩⟩) ⟨52206⟩ 97198)

def event97231 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53108⟩⟩, .relation 97230 0, ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (-1)⟩)

def exact97232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (-1)⟩]

theorem exact97232RawTermsValid :
    exact97232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53108⟩⟩) exact97232RawTerms .large 97227 .exactZero (none)

def event97233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51256⟩⟩) 0 ⟨50929⟩ 97190

def event97234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51256⟩⟩) (.authority (.programFamilyFact))

def exact97235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], []⟩, (1)⟩]

theorem exact97235RawTermsValid :
    exact97235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51256⟩⟩) exact97235RawTerms (.finite 58) 97234 .exactZero (none)

def event97236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51258⟩⟩) 0 ⟨6908⟩ 97212

def event97237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51258⟩⟩) 1 ⟨51256⟩ 97235

def event97238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51258⟩⟩) (.product (.predecessor 0 97236 .coefficient) (.predecessor 1 97237 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51258⟩⟩, .operator (⟨97212, 0⟩, ⟨97235, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact97240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact97240RawTermsValid :
    exact97240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51258⟩⟩) exact97240RawTerms .large 97238 .exactZero (none)

def event97241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 97194

def event97242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact97243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact97243RawTermsValid :
    exact97243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact97243RawTerms .large 97242 .exactZero (none)

def event97244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51259⟩⟩) 0 ⟨7206⟩ 97243

def event97245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51259⟩⟩) 1 ⟨51258⟩ 97240

def event97246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51259⟩⟩) (.sum [.predecessor 0 97244 .coefficient, .predecessor 1 97245 .coefficient])

def exact97247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97247RawTermsValid :
    exact97247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97247 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51259⟩⟩) exact97247RawTerms .large 97246 .exactZero (none)

def event97248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53112⟩⟩) 0 ⟨51259⟩ 97247

def event97249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53112⟩⟩) 1 ⟨53108⟩ 97232

def event97250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53112⟩⟩) (.sum [.predecessor 0 97248 .coefficient, .predecessor 1 97249 .coefficient])

def exact97251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97251RawTermsValid :
    exact97251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53112⟩⟩) exact97251RawTerms .large 97250 .exactZero (none)

def event97252 : Event := .preFoldPolynomial 97251 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact97253RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event97253 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨53112⟩⟩) 97252 exact97253RawTerms .large 97250 .exactZero (none)

def event97254 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50929⟩⟩) ⟨⟨85⟩, ⟨65⟩, ⟨135⟩⟩ ⟨97096, 97254⟩

def event97255 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51859⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩) (1) 0 2 (.universal 97254 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩) (none) 97253)

def event97256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51859⟩⟩, .relation 97255 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩)

def event97257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51859⟩⟩, .relation 97255 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩)

def event97258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51859⟩⟩, .relation 97255 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩)

def event97259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51859⟩⟩, .relation 97255 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact97260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97260RawTermsValid :
    exact97260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51859⟩⟩) exact97260RawTerms .large 97092 (.finite 202072841853861888) (some (97094))

def event97261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53110⟩⟩) 0 ⟨51859⟩ 97260

def event97262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53110⟩⟩) 1 ⟨53109⟩ 97082

def event97263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53110⟩⟩) (.sum [.predecessor 0 97261 .coefficient, .predecessor 1 97262 .coefficient])

def event97264 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53110⟩⟩, .operator (⟨97260, 0⟩, ⟨97082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩, (1)⟩)

def event97265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53110⟩⟩, .operator (⟨97260, 2⟩, ⟨97082, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩, (-1)⟩)

def event97266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53110⟩⟩) (.sum [.result 97260 .summary, .result 97082 .summary])

def exact97267RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact97267RawTermsValid :
    exact97267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53110⟩⟩) exact97267RawTerms .large 97263 (.finite 32189593014266456398474184491008) (some (97266))

def event97268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33144⟩⟩) 0 ⟨31869⟩ 4173

def event97269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33144⟩⟩) (.authority (.programFamilyFact))

def event97270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33144⟩⟩) (.finite 3720)

def event97271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33146⟩⟩) 0 ⟨7177⟩ 15500

def event97272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33146⟩⟩) 1 ⟨33144⟩ 97270

def event97273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33146⟩⟩) (.authority (.operator))

def exact97274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33146⟩⟩]⟩, (1)⟩]

theorem exact97274RawTermsValid :
    exact97274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33146⟩⟩) exact97274RawTerms .large 97273 .exactZero (none)

def event97275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34047⟩⟩) 0 ⟨33146⟩ 97274

def event97276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34047⟩⟩) (.authority (.operator))

def exact97277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩, (1)⟩]

theorem exact97277RawTermsValid :
    exact97277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34047⟩⟩) exact97277RawTerms (.finite 8192) 97276 .exactZero (none)

def event97278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32978⟩⟩) 0 ⟨31622⟩ 4167

def event97279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32978⟩⟩) (.authority (.programFamilyFact))

def eventLeaf6064 : Array AnnotatedEvent := #[
  { event := event97024
    frameStart := 96941 },
  { event := event97025
    frameStart := 96941 },
  { event := event97026
    frameStart := 96941 },
  { event := event97027
    frameStart := 96941 },
  { event := event97028
    frameStart := 96941 },
  { event := event97029
    frameStart := 96941 },
  { event := event97030
    frameStart := 96941 },
  { event := event97031
    frameStart := 96941 },
  { event := event97032
    frameStart := 96941 },
  { event := event97033
    frameStart := 96941 },
  { event := event97034
    frameStart := 96941 },
  { event := event97035
    frameStart := 96941 },
  { event := event97036
    frameStart := 96941 },
  { event := event97037
    frameStart := 96941 },
  { event := event97038
    frameStart := 96941 },
  { event := event97039
    frameStart := 96941 }
]

def eventLeaf6065 : Array AnnotatedEvent := #[
  { event := event97040
    frameStart := 96941 },
  { event := event97041
    frameStart := 96941 },
  { event := event97042
    frameStart := 96941 },
  { event := event97043
    frameStart := 96941 },
  { event := event97044
    frameStart := 96941 },
  { event := event97045
    frameStart := 96941 },
  { event := event97046
    frameStart := 96941 },
  { event := event97047
    frameStart := 96941 },
  { event := event97048
    frameStart := 96941 },
  { event := event97049
    frameStart := 96941 },
  { event := event97050
    frameStart := 96941 },
  { event := event97051
    frameStart := 96941 },
  { event := event97052
    frameStart := 96941 },
  { event := event97053
    frameStart := 96941 },
  { event := event97054
    frameStart := 96941 },
  { event := event97055
    frameStart := 96941 }
]

def eventLeaf6066 : Array AnnotatedEvent := #[
  { event := event97056
    frameStart := 96941 },
  { event := event97057
    frameStart := 96941 },
  { event := event97058
    frameStart := 96941 },
  { event := event97059
    frameStart := 0 },
  { event := event97060
    frameStart := 0 },
  { event := event97061
    frameStart := 0 },
  { event := event97062
    frameStart := 0 },
  { event := event97063
    frameStart := 0 },
  { event := event97064
    frameStart := 0 },
  { event := event97065
    frameStart := 0 },
  { event := event97066
    frameStart := 0 },
  { event := event97067
    frameStart := 0 },
  { event := event97068
    frameStart := 0 },
  { event := event97069
    frameStart := 0 },
  { event := event97070
    frameStart := 0 },
  { event := event97071
    frameStart := 0 }
]

def eventLeaf6067 : Array AnnotatedEvent := #[
  { event := event97072
    frameStart := 0 },
  { event := event97073
    frameStart := 0 },
  { event := event97074
    frameStart := 0 },
  { event := event97075
    frameStart := 0 },
  { event := event97076
    frameStart := 0 },
  { event := event97077
    frameStart := 0 },
  { event := event97078
    frameStart := 0 },
  { event := event97079
    frameStart := 0 },
  { event := event97080
    frameStart := 0 },
  { event := event97081
    frameStart := 0 },
  { event := event97082
    frameStart := 0 },
  { event := event97083
    frameStart := 0 },
  { event := event97084
    frameStart := 0 },
  { event := event97085
    frameStart := 0 },
  { event := event97086
    frameStart := 0 },
  { event := event97087
    frameStart := 0 }
]

def eventLeaf6068 : Array AnnotatedEvent := #[
  { event := event97088
    frameStart := 0 },
  { event := event97089
    frameStart := 0 },
  { event := event97090
    frameStart := 0 },
  { event := event97091
    frameStart := 0 },
  { event := event97092
    frameStart := 0 },
  { event := event97093
    frameStart := 0 },
  { event := event97094
    frameStart := 0 },
  { event := event97095
    frameStart := 0 },
  { event := event97096
    frameStart := 97096 },
  { event := event97097
    frameStart := 97096 },
  { event := event97098
    frameStart := 97096 },
  { event := event97099
    frameStart := 97096 },
  { event := event97100
    frameStart := 97096 },
  { event := event97101
    frameStart := 97096 },
  { event := event97102
    frameStart := 97096 },
  { event := event97103
    frameStart := 97096 }
]

def eventLeaf6069 : Array AnnotatedEvent := #[
  { event := event97104
    frameStart := 97096 },
  { event := event97105
    frameStart := 97096 },
  { event := event97106
    frameStart := 97096 },
  { event := event97107
    frameStart := 97096 },
  { event := event97108
    frameStart := 97096 },
  { event := event97109
    frameStart := 97096 },
  { event := event97110
    frameStart := 97096 },
  { event := event97111
    frameStart := 97096 },
  { event := event97112
    frameStart := 97096 },
  { event := event97113
    frameStart := 97096 },
  { event := event97114
    frameStart := 97096 },
  { event := event97115
    frameStart := 97096 },
  { event := event97116
    frameStart := 97096 },
  { event := event97117
    frameStart := 97096 },
  { event := event97118
    frameStart := 97096 },
  { event := event97119
    frameStart := 97096 }
]

def eventLeaf6070 : Array AnnotatedEvent := #[
  { event := event97120
    frameStart := 97096 },
  { event := event97121
    frameStart := 97096 },
  { event := event97122
    frameStart := 97096 },
  { event := event97123
    frameStart := 97096 },
  { event := event97124
    frameStart := 97096 },
  { event := event97125
    frameStart := 97096 },
  { event := event97126
    frameStart := 97096 },
  { event := event97127
    frameStart := 97096 },
  { event := event97128
    frameStart := 97096 },
  { event := event97129
    frameStart := 97096 },
  { event := event97130
    frameStart := 97096 },
  { event := event97131
    frameStart := 97096 },
  { event := event97132
    frameStart := 97096 },
  { event := event97133
    frameStart := 97096 },
  { event := event97134
    frameStart := 97096 },
  { event := event97135
    frameStart := 97096 }
]

def eventLeaf6071 : Array AnnotatedEvent := #[
  { event := event97136
    frameStart := 97096 },
  { event := event97137
    frameStart := 97096 },
  { event := event97138
    frameStart := 97096 },
  { event := event97139
    frameStart := 97096 },
  { event := event97140
    frameStart := 97096 },
  { event := event97141
    frameStart := 97096 },
  { event := event97142
    frameStart := 97096 },
  { event := event97143
    frameStart := 97096 },
  { event := event97144
    frameStart := 97096 },
  { event := event97145
    frameStart := 97096 },
  { event := event97146
    frameStart := 97096 },
  { event := event97147
    frameStart := 97096 },
  { event := event97148
    frameStart := 97096 },
  { event := event97149
    frameStart := 97096 },
  { event := event97150
    frameStart := 97150 },
  { event := event97151
    frameStart := 97150 }
]

def eventLeaf6072 : Array AnnotatedEvent := #[
  { event := event97152
    frameStart := 97150 },
  { event := event97153
    frameStart := 97150 },
  { event := event97154
    frameStart := 97150 },
  { event := event97155
    frameStart := 97150 },
  { event := event97156
    frameStart := 97150 },
  { event := event97157
    frameStart := 97150 },
  { event := event97158
    frameStart := 97150 },
  { event := event97159
    frameStart := 97150 },
  { event := event97160
    frameStart := 97150 },
  { event := event97161
    frameStart := 97150 },
  { event := event97162
    frameStart := 97150 },
  { event := event97163
    frameStart := 97150 },
  { event := event97164
    frameStart := 97150 },
  { event := event97165
    frameStart := 97150 },
  { event := event97166
    frameStart := 97150 },
  { event := event97167
    frameStart := 97150 }
]

def eventLeaf6073 : Array AnnotatedEvent := #[
  { event := event97168
    frameStart := 97150 },
  { event := event97169
    frameStart := 97150 },
  { event := event97170
    frameStart := 97150 },
  { event := event97171
    frameStart := 97150 },
  { event := event97172
    frameStart := 97150 },
  { event := event97173
    frameStart := 97150 },
  { event := event97174
    frameStart := 97150 },
  { event := event97175
    frameStart := 97150 },
  { event := event97176
    frameStart := 97150 },
  { event := event97177
    frameStart := 97150 },
  { event := event97178
    frameStart := 97150 },
  { event := event97179
    frameStart := 97150 },
  { event := event97180
    frameStart := 97150 },
  { event := event97181
    frameStart := 97150 },
  { event := event97182
    frameStart := 97150 },
  { event := event97183
    frameStart := 97150 }
]

def eventLeaf6074 : Array AnnotatedEvent := #[
  { event := event97184
    frameStart := 97150 },
  { event := event97185
    frameStart := 97150 },
  { event := event97186
    frameStart := 97150 },
  { event := event97187
    frameStart := 97150 },
  { event := event97188
    frameStart := 97150 },
  { event := event97189
    frameStart := 97150 },
  { event := event97190
    frameStart := 97150 },
  { event := event97191
    frameStart := 97150 },
  { event := event97192
    frameStart := 97150 },
  { event := event97193
    frameStart := 97150 },
  { event := event97194
    frameStart := 97150 },
  { event := event97195
    frameStart := 97150 },
  { event := event97196
    frameStart := 97150 },
  { event := event97197
    frameStart := 97150 },
  { event := event97198
    frameStart := 97150 },
  { event := event97199
    frameStart := 97150 }
]

def eventLeaf6075 : Array AnnotatedEvent := #[
  { event := event97200
    frameStart := 97150 },
  { event := event97201
    frameStart := 97150 },
  { event := event97202
    frameStart := 97150 },
  { event := event97203
    frameStart := 97150 },
  { event := event97204
    frameStart := 97150 },
  { event := event97205
    frameStart := 97150 },
  { event := event97206
    frameStart := 97150 },
  { event := event97207
    frameStart := 97150 },
  { event := event97208
    frameStart := 97150 },
  { event := event97209
    frameStart := 97150 },
  { event := event97210
    frameStart := 97150 },
  { event := event97211
    frameStart := 97150 },
  { event := event97212
    frameStart := 97150 },
  { event := event97213
    frameStart := 97150 },
  { event := event97214
    frameStart := 97150 },
  { event := event97215
    frameStart := 97150 }
]

def eventLeaf6076 : Array AnnotatedEvent := #[
  { event := event97216
    frameStart := 97150 },
  { event := event97217
    frameStart := 97150 },
  { event := event97218
    frameStart := 97150 },
  { event := event97219
    frameStart := 97150 },
  { event := event97220
    frameStart := 97150 },
  { event := event97221
    frameStart := 97150 },
  { event := event97222
    frameStart := 97150 },
  { event := event97223
    frameStart := 97150 },
  { event := event97224
    frameStart := 97150 },
  { event := event97225
    frameStart := 97150 },
  { event := event97226
    frameStart := 97150 },
  { event := event97227
    frameStart := 97150 },
  { event := event97228
    frameStart := 97150 },
  { event := event97229
    frameStart := 97150 },
  { event := event97230
    frameStart := 97150 },
  { event := event97231
    frameStart := 97150 }
]

def eventLeaf6077 : Array AnnotatedEvent := #[
  { event := event97232
    frameStart := 97150 },
  { event := event97233
    frameStart := 97150 },
  { event := event97234
    frameStart := 97150 },
  { event := event97235
    frameStart := 97150 },
  { event := event97236
    frameStart := 97150 },
  { event := event97237
    frameStart := 97150 },
  { event := event97238
    frameStart := 97150 },
  { event := event97239
    frameStart := 97150 },
  { event := event97240
    frameStart := 97150 },
  { event := event97241
    frameStart := 97150 },
  { event := event97242
    frameStart := 97150 },
  { event := event97243
    frameStart := 97150 },
  { event := event97244
    frameStart := 97150 },
  { event := event97245
    frameStart := 97150 },
  { event := event97246
    frameStart := 97150 },
  { event := event97247
    frameStart := 97150 }
]

def eventLeaf6078 : Array AnnotatedEvent := #[
  { event := event97248
    frameStart := 97150 },
  { event := event97249
    frameStart := 97150 },
  { event := event97250
    frameStart := 97150 },
  { event := event97251
    frameStart := 97150 },
  { event := event97252
    frameStart := 97150 },
  { event := event97253
    frameStart := 97150 },
  { event := event97254
    frameStart := 0 },
  { event := event97255
    frameStart := 0 },
  { event := event97256
    frameStart := 0 },
  { event := event97257
    frameStart := 0 },
  { event := event97258
    frameStart := 0 },
  { event := event97259
    frameStart := 0 },
  { event := event97260
    frameStart := 0 },
  { event := event97261
    frameStart := 0 },
  { event := event97262
    frameStart := 0 },
  { event := event97263
    frameStart := 0 }
]

def eventLeaf6079 : Array AnnotatedEvent := #[
  { event := event97264
    frameStart := 0 },
  { event := event97265
    frameStart := 0 },
  { event := event97266
    frameStart := 0 },
  { event := event97267
    frameStart := 0 },
  { event := event97268
    frameStart := 0 },
  { event := event97269
    frameStart := 0 },
  { event := event97270
    frameStart := 0 },
  { event := event97271
    frameStart := 0 },
  { event := event97272
    frameStart := 0 },
  { event := event97273
    frameStart := 0 },
  { event := event97274
    frameStart := 0 },
  { event := event97275
    frameStart := 0 },
  { event := event97276
    frameStart := 0 },
  { event := event97277
    frameStart := 0 },
  { event := event97278
    frameStart := 0 },
  { event := event97279
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events379
