import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events836

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event214016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9581⟩⟩) (.scale (.predecessor 0 214014 .coefficient) (.value (.predecessor 1 214015 .coefficient)))

def exact214017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact214017RawTermsValid :
    exact214017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9581⟩⟩) exact214017RawTerms (.finite 8192) 214016 .exactZero (none)

def event214018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7288⟩⟩) 0 ⟨7178⟩ 214007

def event214019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7288⟩⟩) (.identity (.predecessor 0 214018 .coefficient))

def exact214020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact214020RawTermsValid :
    exact214020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7288⟩⟩) exact214020RawTerms .large 214019 .exactZero (none)

def event214021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 0 ⟨7288⟩ 214020

def event214022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 1 ⟨9581⟩ 214017

def event214023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9582⟩⟩) (.product (.predecessor 0 214021 .coefficient) (.predecessor 1 214022 .coefficient) (⟨false, false, none, none, none⟩))

def event214024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9582⟩⟩, .operator (⟨214020, 0⟩, ⟨214017, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact214025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact214025RawTermsValid :
    exact214025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9582⟩⟩) exact214025RawTerms .large 214023 .exactZero (none)

def event214026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52289⟩⟩) 0 ⟨9582⟩ 214025

def event214027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52289⟩⟩) 1 ⟨52288⟩ 214002

def event214028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52289⟩⟩) (.sum [.predecessor 0 214026 .coefficient, .predecessor 1 214027 .coefficient])

def exact214029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214029RawTermsValid :
    exact214029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52289⟩⟩) exact214029RawTerms .large 214028 .exactZero (none)

def event214030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52522⟩⟩) 0 ⟨52289⟩ 214029

def event214031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52522⟩⟩) 1 ⟨52519⟩ 213986

def event214032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52522⟩⟩) (.product (.predecessor 0 214030 .coefficient) (.predecessor 1 214031 .coefficient) (⟨false, false, none, none, none⟩))

def event214033 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52522⟩⟩, .operator (⟨214029, 0⟩, ⟨213986, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (1)⟩)

def event214034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52522⟩⟩, .operator (⟨214029, 1⟩, ⟨213986, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩)

def event214035 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52522⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52519⟩⟩) ⟨52009⟩ 213983)

def event214036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52522⟩⟩, .relation 214035 0, ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (-1)⟩)

def exact214037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (-1)⟩]

theorem exact214037RawTermsValid :
    exact214037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52522⟩⟩) exact214037RawTerms .large 214032 .exactZero (none)

def event214038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50888⟩⟩) 0 ⟨50547⟩ 213975

def event214039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50888⟩⟩) (.authority (.programFamilyFact))

def exact214040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], []⟩, (1)⟩]

theorem exact214040RawTermsValid :
    exact214040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50888⟩⟩) exact214040RawTerms (.finite 10) 214039 .exactZero (none)

def event214041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50890⟩⟩) 0 ⟨6908⟩ 213997

def event214042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50890⟩⟩) 1 ⟨50888⟩ 214040

def event214043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50890⟩⟩) (.product (.predecessor 0 214041 .coefficient) (.predecessor 1 214042 .coefficient) (⟨false, true, none, none, some 1⟩))

def event214044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50890⟩⟩, .operator (⟨213997, 0⟩, ⟨214040, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact214045RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact214045RawTermsValid :
    exact214045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214045 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50890⟩⟩) exact214045RawTerms .large 214043 .exactZero (none)

def event214046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 213979

def event214047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact214048RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact214048RawTermsValid :
    exact214048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact214048RawTerms .large 214047 .exactZero (none)

def event214049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50891⟩⟩) 0 ⟨7183⟩ 214048

def event214050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50891⟩⟩) 1 ⟨50890⟩ 214045

def event214051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50891⟩⟩) (.sum [.predecessor 0 214049 .coefficient, .predecessor 1 214050 .coefficient])

def exact214052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214052RawTermsValid :
    exact214052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50891⟩⟩) exact214052RawTerms .large 214051 .exactZero (none)

def event214053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52523⟩⟩) 0 ⟨50891⟩ 214052

def event214054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52523⟩⟩) 1 ⟨52522⟩ 214037

def event214055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52523⟩⟩) (.sum [.predecessor 0 214053 .coefficient, .predecessor 1 214054 .coefficient])

def exact214056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214056RawTermsValid :
    exact214056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52523⟩⟩) exact214056RawTerms .large 214055 .exactZero (none)

def event214057 : Event := .preFoldPolynomial 214056 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact214058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event214058 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52523⟩⟩) 214057 exact214058RawTerms .large 214055 .exactZero (none)

def event214059 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50547⟩⟩) ⟨⟨62⟩, ⟨40⟩, ⟨135⟩⟩ ⟨213893, 214059⟩

def event214060 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51452⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (1) 0 2 (.universal 214059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51449⟩⟩]⟩) (none) 214058)

def event214061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51452⟩⟩, .relation 214060 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩)

def event214062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51452⟩⟩, .relation 214060 1, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩)

def event214063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51452⟩⟩, .relation 214060 2, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (1)⟩)

def event214064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51452⟩⟩, .relation 214060 3, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact214065RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214065RawTermsValid :
    exact214065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51452⟩⟩) exact214065RawTerms .large 213889 (.finite 202072841853861888) (some (213891))

def event214066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52521⟩⟩) 0 ⟨51452⟩ 214065

def event214067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52521⟩⟩) 1 ⟨52520⟩ 213879

def event214068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52521⟩⟩) (.sum [.predecessor 0 214066 .coefficient, .predecessor 1 214067 .coefficient])

def event214069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52521⟩⟩, .operator (⟨214065, 2⟩, ⟨213879, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], [⟨.program ⟨257⟩, ⟨52009⟩⟩]⟩, (-1)⟩)

def event214070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52521⟩⟩, .operator (⟨214065, 1⟩, ⟨213879, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52519⟩⟩]⟩, (1)⟩)

def event214071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52521⟩⟩) (.sum [.result 214065 .summary, .result 213879 .summary])

def exact214072RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214072RawTermsValid :
    exact214072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52521⟩⟩) exact214072RawTerms .large 214068 (.finite 2997889464187086962688) (some (214071))

def event214073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52954⟩⟩) 0 ⟨52521⟩ 214072

def event214074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52954⟩⟩) 1 ⟨52952⟩ 213795

def event214075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52954⟩⟩) (.product (.predecessor 0 214073 .coefficient) (.predecessor 1 214074 .coefficient) (⟨false, false, none, none, none⟩))

def event214076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52954⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩) [⟨.result 213795 .coefficient, false, none⟩])

def event214077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52954⟩⟩) (.product (.result 214072 .summary) (.transfer 214076) (⟨false, false, none, none, none⟩))

def event214078 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52954⟩⟩, .operator (⟨214072, 0⟩, ⟨213795, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩)

def event214079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52954⟩⟩, .operator (⟨214072, 1⟩, ⟨213795, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩)

def event214080 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52954⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52952⟩⟩) ⟨52161⟩ 213792)

def event214081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52954⟩⟩, .relation 214080 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (-1)⟩)

def exact214082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (-1)⟩]

theorem exact214082RawTermsValid :
    exact214082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52954⟩⟩) exact214082RawTerms .large 214075 (.finite 32189593014266254325632330629120) (some (214077))

def event214083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51756⟩⟩) 0 ⟨50889⟩ 10134

def event214084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51756⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact214085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩]

theorem exact214085RawTermsValid :
    exact214085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51756⟩⟩) exact214085RawTerms (.finite 5647228698) 214084 .exactZero (none)

def event214086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51758⟩⟩) 0 ⟨51756⟩ 214085

def event214087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51758⟩⟩) 1 ⟨2370⟩ 4

def event214088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51758⟩⟩) (.scale (.predecessor 0 214086 .coefficient) (.value (.predecessor 1 214087 .coefficient)))

def exact214089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩]

theorem exact214089RawTermsValid :
    exact214089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214089 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51758⟩⟩) exact214089RawTerms (.finite 5647228698) 214088 .exactZero (none)

def event214090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51759⟩⟩) 0 ⟨5599⟩ 207620

def event214091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51759⟩⟩) 1 ⟨51758⟩ 214089

def event214092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51759⟩⟩) (.product (.predecessor 0 214090 .coefficient) (.predecessor 1 214091 .coefficient) (⟨false, false, none, none, none⟩))

def event214093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51759⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩) [⟨.result 214085 .coefficient, false, none⟩])

def event214094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51759⟩⟩) (.product (.result 207620 .summary) (.transfer 214093) (⟨false, false, none, none, none⟩))

def event214095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51759⟩⟩, .operator (⟨207620, 0⟩, ⟨214089, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩)

def event214096 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51757⟩⟩)

def event214097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event214098 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event214099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.authority (.operator))

def event214100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.finite 6)

def event214101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event214102 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event214103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event214104 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event214105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 214104

def event214106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 214102

def event214107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 214105 .coefficient) (.value (.predecessor 1 214106 .coefficient)))

def event214108 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event214109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 0 ⟨392⟩ 214108

def event214110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 1 ⟨5240⟩ 214100

def event214111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.sum [.predecessor 0 214109 .coefficient, .predecessor 1 214110 .coefficient])

def event214112 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.finite 655346)

def event214113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 0 ⟨5242⟩ 214112

def event214114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 1 ⟨5426⟩ 214098

def event214115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.identity (.predecessor 1 214114 .coefficient))

def event214116 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.finite 655360)

def event214117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24530⟩⟩) 0 ⟨5595⟩ 214116

def event214118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24530⟩⟩) (.authority (.programFamilyFact))

def exact214119RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩], []⟩, (1)⟩]

theorem exact214119RawTermsValid :
    exact214119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24530⟩⟩) exact214119RawTerms (.finite 10) 214118 .exactZero (none)

def event214120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50545⟩⟩) 0 ⟨5595⟩ 214116

def event214121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50545⟩⟩) (.authority (.programFamilyFact))

def exact214122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩, (1)⟩]

theorem exact214122RawTermsValid :
    exact214122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50545⟩⟩) exact214122RawTerms (.finite 10) 214121 .exactZero (none)

def event214123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50546⟩⟩) 0 ⟨50545⟩ 214122

def event214124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50546⟩⟩) 1 ⟨24530⟩ 214119

def event214125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50546⟩⟩) (.product (.predecessor 0 214123 .coefficient) (.predecessor 1 214124 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event214126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50546⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩) [⟨.result 214122 .coefficient, true, some 1⟩, ⟨.result 214119 .coefficient, true, some 1⟩])

def event214127 : Event := .survivorFold (1) 214126

def exact214128RawTerms : List Term := []

theorem exact214128RawTermsValid :
    exact214128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50546⟩⟩) exact214128RawTerms (.finite 100) 214125 (.finite 100) (some (214126))

def event214129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50547⟩⟩) 0 ⟨50546⟩ 214128

def event214130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50547⟩⟩) (.identity (.predecessor 0 214129 .coefficient))

def event214131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50547⟩⟩) (.finite 100)

def event214132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50888⟩⟩) 0 ⟨50547⟩ 214131

def event214133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50888⟩⟩) (.authority (.programFamilyFact))

def exact214134RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], []⟩, (1)⟩]

theorem exact214134RawTermsValid :
    exact214134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50888⟩⟩) exact214134RawTerms (.finite 10) 214133 .exactZero (none)

def event214135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50889⟩⟩) 0 ⟨50888⟩ 214134

def event214136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50889⟩⟩) (.identity (.predecessor 0 214135 .coefficient))

def event214137 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50889⟩⟩) (.finite 10)

def event214138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51756⟩⟩) 0 ⟨50889⟩ 214137

def event214139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51756⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact214140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩]

theorem exact214140RawTermsValid :
    exact214140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51756⟩⟩) exact214140RawTerms (.finite 5647228698) 214139 .exactZero (none)

def event214141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact214142RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact214142RawTermsValid :
    exact214142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214142 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact214142RawTerms .large 214141 .exactZero (none)

def event214143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51757⟩⟩) 0 ⟨35⟩ 214142

def event214144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51757⟩⟩) 1 ⟨51756⟩ 214140

def event214145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51757⟩⟩) (.product (.predecessor 0 214143 .coefficient) (.predecessor 1 214144 .coefficient) (⟨false, false, none, none, none⟩))

def event214146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51757⟩⟩, .operator (⟨214142, 0⟩, ⟨214140, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩)

def exact214147RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩]

theorem exact214147RawTermsValid :
    exact214147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51757⟩⟩) exact214147RawTerms .large 214145 .exactZero (none)

def event214148 : Event := .preFoldPolynomial 214147 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩] .exactZero none

def exact214149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩, (1)⟩]

def event214149 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51757⟩⟩) 214148 exact214149RawTerms .large 214145 .exactZero (none)

def event214150 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨52957⟩⟩)

def event214151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event214152 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event214153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.authority (.operator))

def event214154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5240⟩⟩) (.finite 6)

def event214155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event214156 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event214157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event214158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event214159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 214158

def event214160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 214156

def event214161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 214159 .coefficient) (.value (.predecessor 1 214160 .coefficient)))

def event214162 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event214163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 0 ⟨392⟩ 214162

def event214164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5242⟩⟩) 1 ⟨5240⟩ 214154

def event214165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.sum [.predecessor 0 214163 .coefficient, .predecessor 1 214164 .coefficient])

def event214166 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5242⟩⟩) (.finite 655346)

def event214167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 0 ⟨5242⟩ 214166

def event214168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5595⟩⟩) 1 ⟨5426⟩ 214152

def event214169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.identity (.predecessor 1 214168 .coefficient))

def event214170 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5595⟩⟩) (.finite 655360)

def event214171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24530⟩⟩) 0 ⟨5595⟩ 214170

def event214172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24530⟩⟩) (.authority (.programFamilyFact))

def exact214173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩], []⟩, (1)⟩]

theorem exact214173RawTermsValid :
    exact214173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24530⟩⟩) exact214173RawTerms (.finite 10) 214172 .exactZero (none)

def event214174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50545⟩⟩) 0 ⟨5595⟩ 214170

def event214175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50545⟩⟩) (.authority (.programFamilyFact))

def exact214176RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩, (1)⟩]

theorem exact214176RawTermsValid :
    exact214176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50545⟩⟩) exact214176RawTerms (.finite 10) 214175 .exactZero (none)

def event214177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50546⟩⟩) 0 ⟨50545⟩ 214176

def event214178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50546⟩⟩) 1 ⟨24530⟩ 214173

def event214179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50546⟩⟩) (.product (.predecessor 0 214177 .coefficient) (.predecessor 1 214178 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event214180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50546⟩⟩, .operator (⟨214176, 0⟩, ⟨214173, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩, (1)⟩)

def exact214181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24530⟩⟩, ⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩, (1)⟩]

theorem exact214181RawTermsValid :
    exact214181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50546⟩⟩) exact214181RawTerms (.finite 100) 214179 .exactZero (none)

def event214182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50547⟩⟩) 0 ⟨50546⟩ 214181

def event214183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50547⟩⟩) (.identity (.predecessor 0 214182 .coefficient))

def event214184 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50547⟩⟩) (.finite 100)

def event214185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50888⟩⟩) 0 ⟨50547⟩ 214184

def event214186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50888⟩⟩) (.authority (.programFamilyFact))

def exact214187RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], []⟩, (1)⟩]

theorem exact214187RawTermsValid :
    exact214187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214187 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50888⟩⟩) exact214187RawTerms (.finite 10) 214186 .exactZero (none)

def event214188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50889⟩⟩) 0 ⟨50888⟩ 214187

def event214189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50889⟩⟩) (.identity (.predecessor 0 214188 .coefficient))

def event214190 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50889⟩⟩) (.finite 10)

def event214191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52159⟩⟩) 0 ⟨50889⟩ 214190

def event214192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52159⟩⟩) (.authority (.programFamilyFact))

def event214193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52159⟩⟩) (.finite 3720)

def event214194 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event214195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52161⟩⟩) 0 ⟨7177⟩ 214194

def event214196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52161⟩⟩) 1 ⟨52159⟩ 214193

def event214197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52161⟩⟩) (.authority (.operator))

def exact214198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩]

theorem exact214198RawTermsValid :
    exact214198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52161⟩⟩) exact214198RawTerms .large 214197 .exactZero (none)

def event214199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52952⟩⟩) 0 ⟨52161⟩ 214198

def event214200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52952⟩⟩) (.authority (.operator))

def exact214201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩]

theorem exact214201RawTermsValid :
    exact214201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52952⟩⟩) exact214201RawTerms (.finite 8192) 214200 .exactZero (none)

def event214202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event214203 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event214204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52366⟩⟩) 0 ⟨50889⟩ 214190

def event214205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52366⟩⟩) 1 ⟨136⟩ 214203

def event214206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52366⟩⟩) (.sum [.predecessor 0 214204 .coefficient, .predecessor 1 214205 .coefficient])

def event214207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52366⟩⟩) (.finite 10)

def event214208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52367⟩⟩) 0 ⟨52366⟩ 214207

def event214209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52367⟩⟩) (.identity (.predecessor 0 214208 .coefficient))

def exact214210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], []⟩, (1)⟩]

theorem exact214210RawTermsValid :
    exact214210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52367⟩⟩) exact214210RawTerms (.finite 10) 214209 .exactZero (none)

def event214211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact214212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact214212RawTermsValid :
    exact214212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact214212RawTerms .large 214211 .exactZero (none)

def event214213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52368⟩⟩) 0 ⟨6908⟩ 214212

def event214214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52368⟩⟩) 1 ⟨52367⟩ 214210

def event214215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52368⟩⟩) (.product (.predecessor 0 214213 .coefficient) (.predecessor 1 214214 .coefficient) (⟨false, false, none, none, none⟩))

def event214216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52368⟩⟩, .operator (⟨214212, 0⟩, ⟨214210, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact214217RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact214217RawTermsValid :
    exact214217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52368⟩⟩) exact214217RawTerms .large 214215 .exactZero (none)

def event214218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 214194

def event214219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact214220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact214220RawTermsValid :
    exact214220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact214220RawTerms .large 214219 .exactZero (none)

def event214221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52369⟩⟩) 0 ⟨7183⟩ 214220

def event214222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52369⟩⟩) 1 ⟨52368⟩ 214217

def event214223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52369⟩⟩) (.sum [.predecessor 0 214221 .coefficient, .predecessor 1 214222 .coefficient])

def exact214224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214224RawTermsValid :
    exact214224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52369⟩⟩) exact214224RawTerms .large 214223 .exactZero (none)

def event214225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52953⟩⟩) 0 ⟨52369⟩ 214224

def event214226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52953⟩⟩) 1 ⟨52952⟩ 214201

def event214227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52953⟩⟩) (.product (.predecessor 0 214225 .coefficient) (.predecessor 1 214226 .coefficient) (⟨false, false, none, none, none⟩))

def event214228 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52953⟩⟩, .operator (⟨214224, 0⟩, ⟨214201, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩)

def event214229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52953⟩⟩, .operator (⟨214224, 1⟩, ⟨214201, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩)

def event214230 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52953⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52952⟩⟩) ⟨52161⟩ 214198)

def event214231 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52953⟩⟩, .relation 214230 0, ⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (-1)⟩)

def exact214232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (-1)⟩]

theorem exact214232RawTermsValid :
    exact214232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52953⟩⟩) exact214232RawTerms .large 214227 .exactZero (none)

def event214233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51161⟩⟩) 0 ⟨50889⟩ 214190

def event214234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51161⟩⟩) (.authority (.programFamilyFact))

def exact214235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], []⟩, (1)⟩]

theorem exact214235RawTermsValid :
    exact214235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51161⟩⟩) exact214235RawTerms (.finite 58) 214234 .exactZero (none)

def event214236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51163⟩⟩) 0 ⟨6908⟩ 214212

def event214237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51163⟩⟩) 1 ⟨51161⟩ 214235

def event214238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51163⟩⟩) (.product (.predecessor 0 214236 .coefficient) (.predecessor 1 214237 .coefficient) (⟨false, true, none, none, some 1⟩))

def event214239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51163⟩⟩, .operator (⟨214212, 0⟩, ⟨214235, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact214240RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact214240RawTermsValid :
    exact214240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214240 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51163⟩⟩) exact214240RawTerms .large 214238 .exactZero (none)

def event214241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 214194

def event214242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact214243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact214243RawTermsValid :
    exact214243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact214243RawTerms .large 214242 .exactZero (none)

def event214244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51164⟩⟩) 0 ⟨7206⟩ 214243

def event214245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51164⟩⟩) 1 ⟨51163⟩ 214240

def event214246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51164⟩⟩) (.sum [.predecessor 0 214244 .coefficient, .predecessor 1 214245 .coefficient])

def exact214247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214247RawTermsValid :
    exact214247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214247 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51164⟩⟩) exact214247RawTerms .large 214246 .exactZero (none)

def event214248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52957⟩⟩) 0 ⟨51164⟩ 214247

def event214249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52957⟩⟩) 1 ⟨52953⟩ 214232

def event214250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52957⟩⟩) (.sum [.predecessor 0 214248 .coefficient, .predecessor 1 214249 .coefficient])

def exact214251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214251RawTermsValid :
    exact214251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52957⟩⟩) exact214251RawTerms .large 214250 .exactZero (none)

def event214252 : Event := .preFoldPolynomial 214251 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact214253RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event214253 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52957⟩⟩) 214252 exact214253RawTerms .large 214250 .exactZero (none)

def event214254 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50889⟩⟩) ⟨⟨85⟩, ⟨65⟩, ⟨135⟩⟩ ⟨214096, 214254⟩

def event214255 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51759⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩) (1) 0 2 (.universal 214254 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51756⟩⟩]⟩) (none) 214253)

def event214256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51759⟩⟩, .relation 214255 1, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩)

def event214257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51759⟩⟩, .relation 214255 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩)

def event214258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51759⟩⟩, .relation 214255 2, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩)

def event214259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51759⟩⟩, .relation 214255 3, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact214260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214260RawTermsValid :
    exact214260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51759⟩⟩) exact214260RawTerms .large 214092 (.finite 202072841853861888) (some (214094))

def event214261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52955⟩⟩) 0 ⟨51759⟩ 214260

def event214262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52955⟩⟩) 1 ⟨52954⟩ 214082

def event214263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52955⟩⟩) (.sum [.predecessor 0 214261 .coefficient, .predecessor 1 214262 .coefficient])

def event214264 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52955⟩⟩, .operator (⟨214260, 0⟩, ⟨214082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52952⟩⟩]⟩, (1)⟩)

def event214265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52955⟩⟩, .operator (⟨214260, 2⟩, ⟨214082, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨50888⟩⟩], [⟨.program ⟨257⟩, ⟨52161⟩⟩]⟩, (-1)⟩)

def event214266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52955⟩⟩) (.sum [.result 214260 .summary, .result 214082 .summary])

def exact214267RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact214267RawTermsValid :
    exact214267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event214267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52955⟩⟩) exact214267RawTerms .large 214263 (.finite 32189593014266456398474184491008) (some (214266))

def event214268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33099⟩⟩) 0 ⟨31829⟩ 10157

def event214269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33099⟩⟩) (.authority (.programFamilyFact))

def event214270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33099⟩⟩) (.finite 3720)

def event214271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33101⟩⟩) 0 ⟨7177⟩ 15500

def eventLeaf13376 : Array AnnotatedEvent := #[
  { event := event214016
    frameStart := 213941 },
  { event := event214017
    frameStart := 213941 },
  { event := event214018
    frameStart := 213941 },
  { event := event214019
    frameStart := 213941 },
  { event := event214020
    frameStart := 213941 },
  { event := event214021
    frameStart := 213941 },
  { event := event214022
    frameStart := 213941 },
  { event := event214023
    frameStart := 213941 },
  { event := event214024
    frameStart := 213941 },
  { event := event214025
    frameStart := 213941 },
  { event := event214026
    frameStart := 213941 },
  { event := event214027
    frameStart := 213941 },
  { event := event214028
    frameStart := 213941 },
  { event := event214029
    frameStart := 213941 },
  { event := event214030
    frameStart := 213941 },
  { event := event214031
    frameStart := 213941 }
]

def eventLeaf13377 : Array AnnotatedEvent := #[
  { event := event214032
    frameStart := 213941 },
  { event := event214033
    frameStart := 213941 },
  { event := event214034
    frameStart := 213941 },
  { event := event214035
    frameStart := 213941 },
  { event := event214036
    frameStart := 213941 },
  { event := event214037
    frameStart := 213941 },
  { event := event214038
    frameStart := 213941 },
  { event := event214039
    frameStart := 213941 },
  { event := event214040
    frameStart := 213941 },
  { event := event214041
    frameStart := 213941 },
  { event := event214042
    frameStart := 213941 },
  { event := event214043
    frameStart := 213941 },
  { event := event214044
    frameStart := 213941 },
  { event := event214045
    frameStart := 213941 },
  { event := event214046
    frameStart := 213941 },
  { event := event214047
    frameStart := 213941 }
]

def eventLeaf13378 : Array AnnotatedEvent := #[
  { event := event214048
    frameStart := 213941 },
  { event := event214049
    frameStart := 213941 },
  { event := event214050
    frameStart := 213941 },
  { event := event214051
    frameStart := 213941 },
  { event := event214052
    frameStart := 213941 },
  { event := event214053
    frameStart := 213941 },
  { event := event214054
    frameStart := 213941 },
  { event := event214055
    frameStart := 213941 },
  { event := event214056
    frameStart := 213941 },
  { event := event214057
    frameStart := 213941 },
  { event := event214058
    frameStart := 213941 },
  { event := event214059
    frameStart := 0 },
  { event := event214060
    frameStart := 0 },
  { event := event214061
    frameStart := 0 },
  { event := event214062
    frameStart := 0 },
  { event := event214063
    frameStart := 0 }
]

def eventLeaf13379 : Array AnnotatedEvent := #[
  { event := event214064
    frameStart := 0 },
  { event := event214065
    frameStart := 0 },
  { event := event214066
    frameStart := 0 },
  { event := event214067
    frameStart := 0 },
  { event := event214068
    frameStart := 0 },
  { event := event214069
    frameStart := 0 },
  { event := event214070
    frameStart := 0 },
  { event := event214071
    frameStart := 0 },
  { event := event214072
    frameStart := 0 },
  { event := event214073
    frameStart := 0 },
  { event := event214074
    frameStart := 0 },
  { event := event214075
    frameStart := 0 },
  { event := event214076
    frameStart := 0 },
  { event := event214077
    frameStart := 0 },
  { event := event214078
    frameStart := 0 },
  { event := event214079
    frameStart := 0 }
]

def eventLeaf13380 : Array AnnotatedEvent := #[
  { event := event214080
    frameStart := 0 },
  { event := event214081
    frameStart := 0 },
  { event := event214082
    frameStart := 0 },
  { event := event214083
    frameStart := 0 },
  { event := event214084
    frameStart := 0 },
  { event := event214085
    frameStart := 0 },
  { event := event214086
    frameStart := 0 },
  { event := event214087
    frameStart := 0 },
  { event := event214088
    frameStart := 0 },
  { event := event214089
    frameStart := 0 },
  { event := event214090
    frameStart := 0 },
  { event := event214091
    frameStart := 0 },
  { event := event214092
    frameStart := 0 },
  { event := event214093
    frameStart := 0 },
  { event := event214094
    frameStart := 0 },
  { event := event214095
    frameStart := 0 }
]

def eventLeaf13381 : Array AnnotatedEvent := #[
  { event := event214096
    frameStart := 214096 },
  { event := event214097
    frameStart := 214096 },
  { event := event214098
    frameStart := 214096 },
  { event := event214099
    frameStart := 214096 },
  { event := event214100
    frameStart := 214096 },
  { event := event214101
    frameStart := 214096 },
  { event := event214102
    frameStart := 214096 },
  { event := event214103
    frameStart := 214096 },
  { event := event214104
    frameStart := 214096 },
  { event := event214105
    frameStart := 214096 },
  { event := event214106
    frameStart := 214096 },
  { event := event214107
    frameStart := 214096 },
  { event := event214108
    frameStart := 214096 },
  { event := event214109
    frameStart := 214096 },
  { event := event214110
    frameStart := 214096 },
  { event := event214111
    frameStart := 214096 }
]

def eventLeaf13382 : Array AnnotatedEvent := #[
  { event := event214112
    frameStart := 214096 },
  { event := event214113
    frameStart := 214096 },
  { event := event214114
    frameStart := 214096 },
  { event := event214115
    frameStart := 214096 },
  { event := event214116
    frameStart := 214096 },
  { event := event214117
    frameStart := 214096 },
  { event := event214118
    frameStart := 214096 },
  { event := event214119
    frameStart := 214096 },
  { event := event214120
    frameStart := 214096 },
  { event := event214121
    frameStart := 214096 },
  { event := event214122
    frameStart := 214096 },
  { event := event214123
    frameStart := 214096 },
  { event := event214124
    frameStart := 214096 },
  { event := event214125
    frameStart := 214096 },
  { event := event214126
    frameStart := 214096 },
  { event := event214127
    frameStart := 214096 }
]

def eventLeaf13383 : Array AnnotatedEvent := #[
  { event := event214128
    frameStart := 214096 },
  { event := event214129
    frameStart := 214096 },
  { event := event214130
    frameStart := 214096 },
  { event := event214131
    frameStart := 214096 },
  { event := event214132
    frameStart := 214096 },
  { event := event214133
    frameStart := 214096 },
  { event := event214134
    frameStart := 214096 },
  { event := event214135
    frameStart := 214096 },
  { event := event214136
    frameStart := 214096 },
  { event := event214137
    frameStart := 214096 },
  { event := event214138
    frameStart := 214096 },
  { event := event214139
    frameStart := 214096 },
  { event := event214140
    frameStart := 214096 },
  { event := event214141
    frameStart := 214096 },
  { event := event214142
    frameStart := 214096 },
  { event := event214143
    frameStart := 214096 }
]

def eventLeaf13384 : Array AnnotatedEvent := #[
  { event := event214144
    frameStart := 214096 },
  { event := event214145
    frameStart := 214096 },
  { event := event214146
    frameStart := 214096 },
  { event := event214147
    frameStart := 214096 },
  { event := event214148
    frameStart := 214096 },
  { event := event214149
    frameStart := 214096 },
  { event := event214150
    frameStart := 214150 },
  { event := event214151
    frameStart := 214150 },
  { event := event214152
    frameStart := 214150 },
  { event := event214153
    frameStart := 214150 },
  { event := event214154
    frameStart := 214150 },
  { event := event214155
    frameStart := 214150 },
  { event := event214156
    frameStart := 214150 },
  { event := event214157
    frameStart := 214150 },
  { event := event214158
    frameStart := 214150 },
  { event := event214159
    frameStart := 214150 }
]

def eventLeaf13385 : Array AnnotatedEvent := #[
  { event := event214160
    frameStart := 214150 },
  { event := event214161
    frameStart := 214150 },
  { event := event214162
    frameStart := 214150 },
  { event := event214163
    frameStart := 214150 },
  { event := event214164
    frameStart := 214150 },
  { event := event214165
    frameStart := 214150 },
  { event := event214166
    frameStart := 214150 },
  { event := event214167
    frameStart := 214150 },
  { event := event214168
    frameStart := 214150 },
  { event := event214169
    frameStart := 214150 },
  { event := event214170
    frameStart := 214150 },
  { event := event214171
    frameStart := 214150 },
  { event := event214172
    frameStart := 214150 },
  { event := event214173
    frameStart := 214150 },
  { event := event214174
    frameStart := 214150 },
  { event := event214175
    frameStart := 214150 }
]

def eventLeaf13386 : Array AnnotatedEvent := #[
  { event := event214176
    frameStart := 214150 },
  { event := event214177
    frameStart := 214150 },
  { event := event214178
    frameStart := 214150 },
  { event := event214179
    frameStart := 214150 },
  { event := event214180
    frameStart := 214150 },
  { event := event214181
    frameStart := 214150 },
  { event := event214182
    frameStart := 214150 },
  { event := event214183
    frameStart := 214150 },
  { event := event214184
    frameStart := 214150 },
  { event := event214185
    frameStart := 214150 },
  { event := event214186
    frameStart := 214150 },
  { event := event214187
    frameStart := 214150 },
  { event := event214188
    frameStart := 214150 },
  { event := event214189
    frameStart := 214150 },
  { event := event214190
    frameStart := 214150 },
  { event := event214191
    frameStart := 214150 }
]

def eventLeaf13387 : Array AnnotatedEvent := #[
  { event := event214192
    frameStart := 214150 },
  { event := event214193
    frameStart := 214150 },
  { event := event214194
    frameStart := 214150 },
  { event := event214195
    frameStart := 214150 },
  { event := event214196
    frameStart := 214150 },
  { event := event214197
    frameStart := 214150 },
  { event := event214198
    frameStart := 214150 },
  { event := event214199
    frameStart := 214150 },
  { event := event214200
    frameStart := 214150 },
  { event := event214201
    frameStart := 214150 },
  { event := event214202
    frameStart := 214150 },
  { event := event214203
    frameStart := 214150 },
  { event := event214204
    frameStart := 214150 },
  { event := event214205
    frameStart := 214150 },
  { event := event214206
    frameStart := 214150 },
  { event := event214207
    frameStart := 214150 }
]

def eventLeaf13388 : Array AnnotatedEvent := #[
  { event := event214208
    frameStart := 214150 },
  { event := event214209
    frameStart := 214150 },
  { event := event214210
    frameStart := 214150 },
  { event := event214211
    frameStart := 214150 },
  { event := event214212
    frameStart := 214150 },
  { event := event214213
    frameStart := 214150 },
  { event := event214214
    frameStart := 214150 },
  { event := event214215
    frameStart := 214150 },
  { event := event214216
    frameStart := 214150 },
  { event := event214217
    frameStart := 214150 },
  { event := event214218
    frameStart := 214150 },
  { event := event214219
    frameStart := 214150 },
  { event := event214220
    frameStart := 214150 },
  { event := event214221
    frameStart := 214150 },
  { event := event214222
    frameStart := 214150 },
  { event := event214223
    frameStart := 214150 }
]

def eventLeaf13389 : Array AnnotatedEvent := #[
  { event := event214224
    frameStart := 214150 },
  { event := event214225
    frameStart := 214150 },
  { event := event214226
    frameStart := 214150 },
  { event := event214227
    frameStart := 214150 },
  { event := event214228
    frameStart := 214150 },
  { event := event214229
    frameStart := 214150 },
  { event := event214230
    frameStart := 214150 },
  { event := event214231
    frameStart := 214150 },
  { event := event214232
    frameStart := 214150 },
  { event := event214233
    frameStart := 214150 },
  { event := event214234
    frameStart := 214150 },
  { event := event214235
    frameStart := 214150 },
  { event := event214236
    frameStart := 214150 },
  { event := event214237
    frameStart := 214150 },
  { event := event214238
    frameStart := 214150 },
  { event := event214239
    frameStart := 214150 }
]

def eventLeaf13390 : Array AnnotatedEvent := #[
  { event := event214240
    frameStart := 214150 },
  { event := event214241
    frameStart := 214150 },
  { event := event214242
    frameStart := 214150 },
  { event := event214243
    frameStart := 214150 },
  { event := event214244
    frameStart := 214150 },
  { event := event214245
    frameStart := 214150 },
  { event := event214246
    frameStart := 214150 },
  { event := event214247
    frameStart := 214150 },
  { event := event214248
    frameStart := 214150 },
  { event := event214249
    frameStart := 214150 },
  { event := event214250
    frameStart := 214150 },
  { event := event214251
    frameStart := 214150 },
  { event := event214252
    frameStart := 214150 },
  { event := event214253
    frameStart := 214150 },
  { event := event214254
    frameStart := 0 },
  { event := event214255
    frameStart := 0 }
]

def eventLeaf13391 : Array AnnotatedEvent := #[
  { event := event214256
    frameStart := 0 },
  { event := event214257
    frameStart := 0 },
  { event := event214258
    frameStart := 0 },
  { event := event214259
    frameStart := 0 },
  { event := event214260
    frameStart := 0 },
  { event := event214261
    frameStart := 0 },
  { event := event214262
    frameStart := 0 },
  { event := event214263
    frameStart := 0 },
  { event := event214264
    frameStart := 0 },
  { event := event214265
    frameStart := 0 },
  { event := event214266
    frameStart := 0 },
  { event := event214267
    frameStart := 0 },
  { event := event214268
    frameStart := 0 },
  { event := event214269
    frameStart := 0 },
  { event := event214270
    frameStart := 0 },
  { event := event214271
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events836
