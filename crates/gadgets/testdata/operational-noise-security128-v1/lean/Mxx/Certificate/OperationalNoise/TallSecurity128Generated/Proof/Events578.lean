import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events578

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event147968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event147969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event147970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event147971 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event147972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event147973 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event147974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 147973

def event147975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 147971

def event147976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 147974 .coefficient) (.value (.predecessor 1 147975 .coefficient)))

def event147977 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event147978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 147977

def event147979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 147969

def event147980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 147978 .coefficient, .predecessor 1 147979 .coefficient])

def event147981 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event147982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 147981

def event147983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 147967

def event147984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 147983 .coefficient))

def event147985 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event147986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21326⟩⟩) 0 ⟨5469⟩ 147985

def event147987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21326⟩⟩) (.authority (.programFamilyFact))

def exact147988RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩, (1)⟩]

theorem exact147988RawTermsValid :
    exact147988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event147988 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21326⟩⟩) exact147988RawTerms (.finite 4) 147987 .exactZero (none)

def event147989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20996⟩⟩) 0 ⟨5469⟩ 147985

def event147990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20996⟩⟩) (.authority (.programFamilyFact))

def exact147991RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩], []⟩, (1)⟩]

theorem exact147991RawTermsValid :
    exact147991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event147991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20996⟩⟩) exact147991RawTerms (.finite 4) 147990 .exactZero (none)

def event147992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21327⟩⟩) 0 ⟨20996⟩ 147991

def event147993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21327⟩⟩) 1 ⟨21326⟩ 147988

def event147994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21327⟩⟩) (.product (.predecessor 0 147992 .coefficient) (.predecessor 1 147993 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event147995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21327⟩⟩, .operator (⟨147991, 0⟩, ⟨147988, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩, (1)⟩)

def exact147996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩, (1)⟩]

theorem exact147996RawTermsValid :
    exact147996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event147996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21327⟩⟩) exact147996RawTerms (.finite 16) 147994 .exactZero (none)

def event147997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21328⟩⟩) 0 ⟨21327⟩ 147996

def event147998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21328⟩⟩) (.identity (.predecessor 0 147997 .coefficient))

def event147999 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21328⟩⟩) (.finite 16)

def event148000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21752⟩⟩) 0 ⟨21328⟩ 147999

def event148001 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21752⟩⟩) (.authority (.programFamilyFact))

def exact148002RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], []⟩, (1)⟩]

theorem exact148002RawTermsValid :
    exact148002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148002 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21752⟩⟩) exact148002RawTerms (.finite 4) 148001 .exactZero (none)

def event148003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21753⟩⟩) 0 ⟨21752⟩ 148002

def event148004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21753⟩⟩) (.identity (.predecessor 0 148003 .coefficient))

def event148005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21753⟩⟩) (.finite 4)

def event148006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23016⟩⟩) 0 ⟨21753⟩ 148005

def event148007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23016⟩⟩) (.authority (.programFamilyFact))

def event148008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23016⟩⟩) (.finite 3720)

def event148009 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event148010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23017⟩⟩) 0 ⟨7177⟩ 148009

def event148011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23017⟩⟩) 1 ⟨23016⟩ 148008

def event148012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23017⟩⟩) (.authority (.operator))

def exact148013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩]

theorem exact148013RawTermsValid :
    exact148013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23017⟩⟩) exact148013RawTerms .large 148012 .exactZero (none)

def event148014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23648⟩⟩) 0 ⟨23017⟩ 148013

def event148015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23648⟩⟩) (.authority (.operator))

def exact148016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (1)⟩]

theorem exact148016RawTermsValid :
    exact148016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23648⟩⟩) exact148016RawTerms (.finite 8192) 148015 .exactZero (none)

def event148017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event148018 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event148019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23258⟩⟩) 0 ⟨21753⟩ 148005

def event148020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23258⟩⟩) 1 ⟨136⟩ 148018

def event148021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23258⟩⟩) (.sum [.predecessor 0 148019 .coefficient, .predecessor 1 148020 .coefficient])

def event148022 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23258⟩⟩) (.finite 4)

def event148023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23259⟩⟩) 0 ⟨23258⟩ 148022

def event148024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23259⟩⟩) (.identity (.predecessor 0 148023 .coefficient))

def exact148025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], []⟩, (1)⟩]

theorem exact148025RawTermsValid :
    exact148025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23259⟩⟩) exact148025RawTerms (.finite 4) 148024 .exactZero (none)

def event148026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact148027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact148027RawTermsValid :
    exact148027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact148027RawTerms .large 148026 .exactZero (none)

def event148028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23260⟩⟩) 0 ⟨6908⟩ 148027

def event148029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23260⟩⟩) 1 ⟨23259⟩ 148025

def event148030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23260⟩⟩) (.product (.predecessor 0 148028 .coefficient) (.predecessor 1 148029 .coefficient) (⟨false, false, none, none, none⟩))

def event148031 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23260⟩⟩, .operator (⟨148027, 0⟩, ⟨148025, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact148032RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact148032RawTermsValid :
    exact148032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23260⟩⟩) exact148032RawTerms .large 148030 .exactZero (none)

def event148033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 148009

def event148034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact148035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact148035RawTermsValid :
    exact148035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact148035RawTerms .large 148034 .exactZero (none)

def event148036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23261⟩⟩) 0 ⟨7181⟩ 148035

def event148037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23261⟩⟩) 1 ⟨23260⟩ 148032

def event148038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23261⟩⟩) (.sum [.predecessor 0 148036 .coefficient, .predecessor 1 148037 .coefficient])

def exact148039RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148039RawTermsValid :
    exact148039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148039 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23261⟩⟩) exact148039RawTerms .large 148038 .exactZero (none)

def event148040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23649⟩⟩) 0 ⟨23261⟩ 148039

def event148041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23649⟩⟩) 1 ⟨23648⟩ 148016

def event148042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23649⟩⟩) (.product (.predecessor 0 148040 .coefficient) (.predecessor 1 148041 .coefficient) (⟨false, false, none, none, none⟩))

def event148043 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23649⟩⟩, .operator (⟨148039, 0⟩, ⟨148016, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (1)⟩)

def event148044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23649⟩⟩, .operator (⟨148039, 1⟩, ⟨148016, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩)

def event148045 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23649⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23648⟩⟩) ⟨23017⟩ 148013)

def event148046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23649⟩⟩, .relation 148045 0, ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (-1)⟩)

def exact148047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (-1)⟩]

theorem exact148047RawTermsValid :
    exact148047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23649⟩⟩) exact148047RawTerms .large 148042 .exactZero (none)

def event148048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21948⟩⟩) 0 ⟨21753⟩ 148005

def event148049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21948⟩⟩) (.authority (.programFamilyFact))

def exact148050RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩]

theorem exact148050RawTermsValid :
    exact148050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21948⟩⟩) exact148050RawTerms (.finite 4) 148049 .exactZero (none)

def event148051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21951⟩⟩) 0 ⟨6908⟩ 148027

def event148052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21951⟩⟩) 1 ⟨21948⟩ 148050

def event148053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21951⟩⟩) (.product (.predecessor 0 148051 .coefficient) (.predecessor 1 148052 .coefficient) (⟨false, true, none, none, some 1⟩))

def event148054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21951⟩⟩, .operator (⟨148027, 0⟩, ⟨148050, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact148055RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact148055RawTermsValid :
    exact148055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21951⟩⟩) exact148055RawTerms .large 148053 .exactZero (none)

def event148056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7201⟩⟩) 0 ⟨7177⟩ 148009

def event148057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7201⟩⟩) (.authority (.operator))

def exact148058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩]

theorem exact148058RawTermsValid :
    exact148058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7201⟩⟩) exact148058RawTerms .large 148057 .exactZero (none)

def event148059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21952⟩⟩) 0 ⟨7201⟩ 148058

def event148060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21952⟩⟩) 1 ⟨21951⟩ 148055

def event148061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21952⟩⟩) (.sum [.predecessor 0 148059 .coefficient, .predecessor 1 148060 .coefficient])

def exact148062RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148062RawTermsValid :
    exact148062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148062 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21952⟩⟩) exact148062RawTerms .large 148061 .exactZero (none)

def event148063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23654⟩⟩) 0 ⟨21952⟩ 148062

def event148064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23654⟩⟩) 1 ⟨23649⟩ 148047

def event148065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23654⟩⟩) (.sum [.predecessor 0 148063 .coefficient, .predecessor 1 148064 .coefficient])

def exact148066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148066RawTermsValid :
    exact148066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23654⟩⟩) exact148066RawTerms .large 148065 .exactZero (none)

def event148067 : Event := .preFoldPolynomial 148066 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact148068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event148068 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨23654⟩⟩) 148067 exact148068RawTerms .large 148065 .exactZero (none)

def event148069 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21753⟩⟩) ⟨⟨80⟩, ⟨60⟩, ⟨135⟩⟩ ⟨147911, 148069⟩

def event148070 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22535⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22532⟩⟩]⟩) (1) 0 2 (.universal 148069 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22532⟩⟩]⟩) (none) 148068)

def event148071 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22535⟩⟩, .relation 148070 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩)

def event148072 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22535⟩⟩, .relation 148070 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩)

def event148073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22535⟩⟩, .relation 148070 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩)

def event148074 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22535⟩⟩, .relation 148070 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact148075RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148075RawTermsValid :
    exact148075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22535⟩⟩) exact148075RawTerms .large 147907 (.finite 202072841853861888) (some (147909))

def event148076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23651⟩⟩) 0 ⟨22535⟩ 148075

def event148077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23651⟩⟩) 1 ⟨23650⟩ 147897

def event148078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23651⟩⟩) (.sum [.predecessor 0 148076 .coefficient, .predecessor 1 148077 .coefficient])

def event148079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23651⟩⟩, .operator (⟨148075, 0⟩, ⟨147897, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23648⟩⟩]⟩, (1)⟩)

def event148080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23651⟩⟩, .operator (⟨148075, 2⟩, ⟨147897, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23017⟩⟩]⟩, (-1)⟩)

def event148081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23651⟩⟩) (.sum [.result 148075 .summary, .result 147897 .summary])

def exact148082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148082RawTermsValid :
    exact148082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23651⟩⟩) exact148082RawTerms .large 148078 (.finite 32189003662929394266751515230208) (some (148081))

def event148083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23652⟩⟩) 0 ⟨23651⟩ 148082

def event148084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23652⟩⟩) 1 ⟨7156⟩ 15842

def event148085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23652⟩⟩) (.product (.predecessor 0 148083 .coefficient) (.predecessor 1 148084 .coefficient) (⟨false, false, none, none, none⟩))

def event148086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23652⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) [⟨.result 15838 .coefficient, false, none⟩])

def event148087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23652⟩⟩) (.product (.result 148082 .summary) (.transfer 148086) (⟨false, false, none, none, none⟩))

def event148088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23652⟩⟩, .operator (⟨148082, 0⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩)

def event148089 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23652⟩⟩, .operator (⟨148082, 1⟩, ⟨15842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event148090 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23652⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835)

def event148091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23652⟩⟩, .relation 148090 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact148092RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact148092RawTermsValid :
    exact148092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23652⟩⟩) exact148092RawTerms .large 148085 (.finite 345626795057764889831969145180473178193920) (some (148087))

def event148093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19797⟩⟩) 0 ⟨7177⟩ 15500

def event148094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19797⟩⟩) 1 ⟨19796⟩ 142109

def event148095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19797⟩⟩) (.authority (.operator))

def exact148096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19797⟩⟩]⟩, (1)⟩]

theorem exact148096RawTermsValid :
    exact148096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148096 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19797⟩⟩) exact148096RawTerms .large 148095 .exactZero (none)

def event148097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20428⟩⟩) 0 ⟨19797⟩ 148096

def event148098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20428⟩⟩) (.authority (.operator))

def exact148099RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩, (1)⟩]

theorem exact148099RawTermsValid :
    exact148099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148099 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20428⟩⟩) exact148099RawTerms (.finite 8192) 148098 .exactZero (none)

def event148100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20430⟩⟩) 0 ⟨20144⟩ 142393

def event148101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20430⟩⟩) 1 ⟨20428⟩ 148099

def event148102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20430⟩⟩) (.product (.predecessor 0 148100 .coefficient) (.predecessor 1 148101 .coefficient) (⟨false, false, none, none, none⟩))

def event148103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20430⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩) [⟨.result 148099 .coefficient, false, none⟩])

def event148104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20430⟩⟩) (.product (.result 142393 .summary) (.transfer 148103) (⟨false, false, none, none, none⟩))

def event148105 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20430⟩⟩, .operator (⟨142393, 0⟩, ⟨148099, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩, (1)⟩)

def event148106 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20430⟩⟩, .operator (⟨142393, 1⟩, ⟨148099, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩, (-1)⟩)

def event148107 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20430⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20428⟩⟩) ⟨19797⟩ 148096)

def event148108 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20430⟩⟩, .relation 148107 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19797⟩⟩]⟩, (-1)⟩)

def exact148109RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20428⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨18532⟩⟩], [⟨.program ⟨257⟩, ⟨19797⟩⟩]⟩, (-1)⟩]

theorem exact148109RawTermsValid :
    exact148109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20430⟩⟩) exact148109RawTerms .large 148102 (.finite 32188905437706348505289216491520) (some (148104))

def event148110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19312⟩⟩) 0 ⟨18533⟩ 6463

def event148111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19312⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact148112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact148112RawTermsValid :
    exact148112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19312⟩⟩) exact148112RawTerms (.finite 5647228698) 148111 .exactZero (none)

def event148113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19314⟩⟩) 0 ⟨19312⟩ 148112

def event148114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19314⟩⟩) 1 ⟨2370⟩ 4

def event148115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19314⟩⟩) (.scale (.predecessor 0 148113 .coefficient) (.value (.predecessor 1 148114 .coefficient)))

def exact148116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact148116RawTermsValid :
    exact148116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148116 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19314⟩⟩) exact148116RawTerms (.finite 5647228698) 148115 .exactZero (none)

def event148117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19315⟩⟩) 0 ⟨5473⟩ 134495

def event148118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19315⟩⟩) 1 ⟨19314⟩ 148116

def event148119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19315⟩⟩) (.product (.predecessor 0 148117 .coefficient) (.predecessor 1 148118 .coefficient) (⟨false, false, none, none, none⟩))

def event148120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19315⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩) [⟨.result 148112 .coefficient, false, none⟩])

def event148121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19315⟩⟩) (.product (.result 134495 .summary) (.transfer 148120) (⟨false, false, none, none, none⟩))

def event148122 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19315⟩⟩, .operator (⟨134495, 0⟩, ⟨148116, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩)

def event148123 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19313⟩⟩)

def event148124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event148125 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event148126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event148127 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event148128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event148129 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event148130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event148131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event148132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 148131

def event148133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 148129

def event148134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 148132 .coefficient) (.value (.predecessor 1 148133 .coefficient)))

def event148135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event148136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 148135

def event148137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 148127

def event148138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 148136 .coefficient, .predecessor 1 148137 .coefficient])

def event148139 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event148140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 148139

def event148141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 148125

def event148142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 148141 .coefficient))

def event148143 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event148144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18106⟩⟩) 0 ⟨5469⟩ 148143

def event148145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18106⟩⟩) (.authority (.programFamilyFact))

def exact148146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩, (1)⟩]

theorem exact148146RawTermsValid :
    exact148146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18106⟩⟩) exact148146RawTerms (.finite 3) 148145 .exactZero (none)

def event148147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12576⟩⟩) 0 ⟨5469⟩ 148143

def event148148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12576⟩⟩) (.authority (.programFamilyFact))

def exact148149RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩], []⟩, (1)⟩]

theorem exact148149RawTermsValid :
    exact148149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12576⟩⟩) exact148149RawTerms (.finite 3) 148148 .exactZero (none)

def event148150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 0 ⟨12576⟩ 148149

def event148151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 1 ⟨18106⟩ 148146

def event148152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18107⟩⟩) (.product (.predecessor 0 148150 .coefficient) (.predecessor 1 148151 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event148153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18107⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩) [⟨.result 148149 .coefficient, true, some 1⟩, ⟨.result 148146 .coefficient, true, some 1⟩])

def event148154 : Event := .survivorFold (1) 148153

def exact148155RawTerms : List Term := []

theorem exact148155RawTermsValid :
    exact148155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18107⟩⟩) exact148155RawTerms (.finite 9) 148152 (.finite 9) (some (148153))

def event148156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18108⟩⟩) 0 ⟨18107⟩ 148155

def event148157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.identity (.predecessor 0 148156 .coefficient))

def event148158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.finite 9)

def event148159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18532⟩⟩) 0 ⟨18108⟩ 148158

def event148160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18532⟩⟩) (.authority (.programFamilyFact))

def exact148161RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], []⟩, (1)⟩]

theorem exact148161RawTermsValid :
    exact148161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18532⟩⟩) exact148161RawTerms (.finite 3) 148160 .exactZero (none)

def event148162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18533⟩⟩) 0 ⟨18532⟩ 148161

def event148163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.identity (.predecessor 0 148162 .coefficient))

def event148164 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.finite 3)

def event148165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19312⟩⟩) 0 ⟨18533⟩ 148164

def event148166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19312⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact148167RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact148167RawTermsValid :
    exact148167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148167 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19312⟩⟩) exact148167RawTerms (.finite 5647228698) 148166 .exactZero (none)

def event148168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact148169RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact148169RawTermsValid :
    exact148169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact148169RawTerms .large 148168 .exactZero (none)

def event148170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19313⟩⟩) 0 ⟨35⟩ 148169

def event148171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19313⟩⟩) 1 ⟨19312⟩ 148167

def event148172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19313⟩⟩) (.product (.predecessor 0 148170 .coefficient) (.predecessor 1 148171 .coefficient) (⟨false, false, none, none, none⟩))

def event148173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19313⟩⟩, .operator (⟨148169, 0⟩, ⟨148167, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩)

def exact148174RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact148174RawTermsValid :
    exact148174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19313⟩⟩) exact148174RawTerms .large 148172 .exactZero (none)

def event148175 : Event := .preFoldPolynomial 148174 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩] .exactZero none

def exact148176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19312⟩⟩]⟩, (1)⟩]

def event148176 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨19313⟩⟩) 148175 exact148176RawTerms .large 148172 .exactZero (none)

def event148177 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨20434⟩⟩)

def event148178 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event148179 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event148180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event148181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event148182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event148183 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event148184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event148185 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event148186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 148185

def event148187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 148183

def event148188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 148186 .coefficient) (.value (.predecessor 1 148187 .coefficient)))

def event148189 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event148190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 148189

def event148191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 148181

def event148192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 148190 .coefficient, .predecessor 1 148191 .coefficient])

def event148193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event148194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 148193

def event148195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 148179

def event148196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 148195 .coefficient))

def event148197 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event148198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18106⟩⟩) 0 ⟨5469⟩ 148197

def event148199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18106⟩⟩) (.authority (.programFamilyFact))

def exact148200RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩, (1)⟩]

theorem exact148200RawTermsValid :
    exact148200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148200 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18106⟩⟩) exact148200RawTerms (.finite 3) 148199 .exactZero (none)

def event148201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12576⟩⟩) 0 ⟨5469⟩ 148197

def event148202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12576⟩⟩) (.authority (.programFamilyFact))

def exact148203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩], []⟩, (1)⟩]

theorem exact148203RawTermsValid :
    exact148203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12576⟩⟩) exact148203RawTerms (.finite 3) 148202 .exactZero (none)

def event148204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 0 ⟨12576⟩ 148203

def event148205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 1 ⟨18106⟩ 148200

def event148206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18107⟩⟩) (.product (.predecessor 0 148204 .coefficient) (.predecessor 1 148205 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event148207 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18107⟩⟩, .operator (⟨148203, 0⟩, ⟨148200, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩, (1)⟩)

def exact148208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩, (1)⟩]

theorem exact148208RawTermsValid :
    exact148208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18107⟩⟩) exact148208RawTerms (.finite 9) 148206 .exactZero (none)

def event148209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18108⟩⟩) 0 ⟨18107⟩ 148208

def event148210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.identity (.predecessor 0 148209 .coefficient))

def event148211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.finite 9)

def event148212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18532⟩⟩) 0 ⟨18108⟩ 148211

def event148213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18532⟩⟩) (.authority (.programFamilyFact))

def exact148214RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], []⟩, (1)⟩]

theorem exact148214RawTermsValid :
    exact148214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event148214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18532⟩⟩) exact148214RawTerms (.finite 3) 148213 .exactZero (none)

def event148215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18533⟩⟩) 0 ⟨18532⟩ 148214

def event148216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.identity (.predecessor 0 148215 .coefficient))

def event148217 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.finite 3)

def event148218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19796⟩⟩) 0 ⟨18533⟩ 148217

def event148219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19796⟩⟩) (.authority (.programFamilyFact))

def event148220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19796⟩⟩) (.finite 3720)

def event148221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event148222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19797⟩⟩) 0 ⟨7177⟩ 148221

def event148223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19797⟩⟩) 1 ⟨19796⟩ 148220

def eventLeaf9248 : Array AnnotatedEvent := #[
  { event := event147968
    frameStart := 147965 },
  { event := event147969
    frameStart := 147965 },
  { event := event147970
    frameStart := 147965 },
  { event := event147971
    frameStart := 147965 },
  { event := event147972
    frameStart := 147965 },
  { event := event147973
    frameStart := 147965 },
  { event := event147974
    frameStart := 147965 },
  { event := event147975
    frameStart := 147965 },
  { event := event147976
    frameStart := 147965 },
  { event := event147977
    frameStart := 147965 },
  { event := event147978
    frameStart := 147965 },
  { event := event147979
    frameStart := 147965 },
  { event := event147980
    frameStart := 147965 },
  { event := event147981
    frameStart := 147965 },
  { event := event147982
    frameStart := 147965 },
  { event := event147983
    frameStart := 147965 }
]

def eventLeaf9249 : Array AnnotatedEvent := #[
  { event := event147984
    frameStart := 147965 },
  { event := event147985
    frameStart := 147965 },
  { event := event147986
    frameStart := 147965 },
  { event := event147987
    frameStart := 147965 },
  { event := event147988
    frameStart := 147965 },
  { event := event147989
    frameStart := 147965 },
  { event := event147990
    frameStart := 147965 },
  { event := event147991
    frameStart := 147965 },
  { event := event147992
    frameStart := 147965 },
  { event := event147993
    frameStart := 147965 },
  { event := event147994
    frameStart := 147965 },
  { event := event147995
    frameStart := 147965 },
  { event := event147996
    frameStart := 147965 },
  { event := event147997
    frameStart := 147965 },
  { event := event147998
    frameStart := 147965 },
  { event := event147999
    frameStart := 147965 }
]

def eventLeaf9250 : Array AnnotatedEvent := #[
  { event := event148000
    frameStart := 147965 },
  { event := event148001
    frameStart := 147965 },
  { event := event148002
    frameStart := 147965 },
  { event := event148003
    frameStart := 147965 },
  { event := event148004
    frameStart := 147965 },
  { event := event148005
    frameStart := 147965 },
  { event := event148006
    frameStart := 147965 },
  { event := event148007
    frameStart := 147965 },
  { event := event148008
    frameStart := 147965 },
  { event := event148009
    frameStart := 147965 },
  { event := event148010
    frameStart := 147965 },
  { event := event148011
    frameStart := 147965 },
  { event := event148012
    frameStart := 147965 },
  { event := event148013
    frameStart := 147965 },
  { event := event148014
    frameStart := 147965 },
  { event := event148015
    frameStart := 147965 }
]

def eventLeaf9251 : Array AnnotatedEvent := #[
  { event := event148016
    frameStart := 147965 },
  { event := event148017
    frameStart := 147965 },
  { event := event148018
    frameStart := 147965 },
  { event := event148019
    frameStart := 147965 },
  { event := event148020
    frameStart := 147965 },
  { event := event148021
    frameStart := 147965 },
  { event := event148022
    frameStart := 147965 },
  { event := event148023
    frameStart := 147965 },
  { event := event148024
    frameStart := 147965 },
  { event := event148025
    frameStart := 147965 },
  { event := event148026
    frameStart := 147965 },
  { event := event148027
    frameStart := 147965 },
  { event := event148028
    frameStart := 147965 },
  { event := event148029
    frameStart := 147965 },
  { event := event148030
    frameStart := 147965 },
  { event := event148031
    frameStart := 147965 }
]

def eventLeaf9252 : Array AnnotatedEvent := #[
  { event := event148032
    frameStart := 147965 },
  { event := event148033
    frameStart := 147965 },
  { event := event148034
    frameStart := 147965 },
  { event := event148035
    frameStart := 147965 },
  { event := event148036
    frameStart := 147965 },
  { event := event148037
    frameStart := 147965 },
  { event := event148038
    frameStart := 147965 },
  { event := event148039
    frameStart := 147965 },
  { event := event148040
    frameStart := 147965 },
  { event := event148041
    frameStart := 147965 },
  { event := event148042
    frameStart := 147965 },
  { event := event148043
    frameStart := 147965 },
  { event := event148044
    frameStart := 147965 },
  { event := event148045
    frameStart := 147965 },
  { event := event148046
    frameStart := 147965 },
  { event := event148047
    frameStart := 147965 }
]

def eventLeaf9253 : Array AnnotatedEvent := #[
  { event := event148048
    frameStart := 147965 },
  { event := event148049
    frameStart := 147965 },
  { event := event148050
    frameStart := 147965 },
  { event := event148051
    frameStart := 147965 },
  { event := event148052
    frameStart := 147965 },
  { event := event148053
    frameStart := 147965 },
  { event := event148054
    frameStart := 147965 },
  { event := event148055
    frameStart := 147965 },
  { event := event148056
    frameStart := 147965 },
  { event := event148057
    frameStart := 147965 },
  { event := event148058
    frameStart := 147965 },
  { event := event148059
    frameStart := 147965 },
  { event := event148060
    frameStart := 147965 },
  { event := event148061
    frameStart := 147965 },
  { event := event148062
    frameStart := 147965 },
  { event := event148063
    frameStart := 147965 }
]

def eventLeaf9254 : Array AnnotatedEvent := #[
  { event := event148064
    frameStart := 147965 },
  { event := event148065
    frameStart := 147965 },
  { event := event148066
    frameStart := 147965 },
  { event := event148067
    frameStart := 147965 },
  { event := event148068
    frameStart := 147965 },
  { event := event148069
    frameStart := 0 },
  { event := event148070
    frameStart := 0 },
  { event := event148071
    frameStart := 0 },
  { event := event148072
    frameStart := 0 },
  { event := event148073
    frameStart := 0 },
  { event := event148074
    frameStart := 0 },
  { event := event148075
    frameStart := 0 },
  { event := event148076
    frameStart := 0 },
  { event := event148077
    frameStart := 0 },
  { event := event148078
    frameStart := 0 },
  { event := event148079
    frameStart := 0 }
]

def eventLeaf9255 : Array AnnotatedEvent := #[
  { event := event148080
    frameStart := 0 },
  { event := event148081
    frameStart := 0 },
  { event := event148082
    frameStart := 0 },
  { event := event148083
    frameStart := 0 },
  { event := event148084
    frameStart := 0 },
  { event := event148085
    frameStart := 0 },
  { event := event148086
    frameStart := 0 },
  { event := event148087
    frameStart := 0 },
  { event := event148088
    frameStart := 0 },
  { event := event148089
    frameStart := 0 },
  { event := event148090
    frameStart := 0 },
  { event := event148091
    frameStart := 0 },
  { event := event148092
    frameStart := 0 },
  { event := event148093
    frameStart := 0 },
  { event := event148094
    frameStart := 0 },
  { event := event148095
    frameStart := 0 }
]

def eventLeaf9256 : Array AnnotatedEvent := #[
  { event := event148096
    frameStart := 0 },
  { event := event148097
    frameStart := 0 },
  { event := event148098
    frameStart := 0 },
  { event := event148099
    frameStart := 0 },
  { event := event148100
    frameStart := 0 },
  { event := event148101
    frameStart := 0 },
  { event := event148102
    frameStart := 0 },
  { event := event148103
    frameStart := 0 },
  { event := event148104
    frameStart := 0 },
  { event := event148105
    frameStart := 0 },
  { event := event148106
    frameStart := 0 },
  { event := event148107
    frameStart := 0 },
  { event := event148108
    frameStart := 0 },
  { event := event148109
    frameStart := 0 },
  { event := event148110
    frameStart := 0 },
  { event := event148111
    frameStart := 0 }
]

def eventLeaf9257 : Array AnnotatedEvent := #[
  { event := event148112
    frameStart := 0 },
  { event := event148113
    frameStart := 0 },
  { event := event148114
    frameStart := 0 },
  { event := event148115
    frameStart := 0 },
  { event := event148116
    frameStart := 0 },
  { event := event148117
    frameStart := 0 },
  { event := event148118
    frameStart := 0 },
  { event := event148119
    frameStart := 0 },
  { event := event148120
    frameStart := 0 },
  { event := event148121
    frameStart := 0 },
  { event := event148122
    frameStart := 0 },
  { event := event148123
    frameStart := 148123 },
  { event := event148124
    frameStart := 148123 },
  { event := event148125
    frameStart := 148123 },
  { event := event148126
    frameStart := 148123 },
  { event := event148127
    frameStart := 148123 }
]

def eventLeaf9258 : Array AnnotatedEvent := #[
  { event := event148128
    frameStart := 148123 },
  { event := event148129
    frameStart := 148123 },
  { event := event148130
    frameStart := 148123 },
  { event := event148131
    frameStart := 148123 },
  { event := event148132
    frameStart := 148123 },
  { event := event148133
    frameStart := 148123 },
  { event := event148134
    frameStart := 148123 },
  { event := event148135
    frameStart := 148123 },
  { event := event148136
    frameStart := 148123 },
  { event := event148137
    frameStart := 148123 },
  { event := event148138
    frameStart := 148123 },
  { event := event148139
    frameStart := 148123 },
  { event := event148140
    frameStart := 148123 },
  { event := event148141
    frameStart := 148123 },
  { event := event148142
    frameStart := 148123 },
  { event := event148143
    frameStart := 148123 }
]

def eventLeaf9259 : Array AnnotatedEvent := #[
  { event := event148144
    frameStart := 148123 },
  { event := event148145
    frameStart := 148123 },
  { event := event148146
    frameStart := 148123 },
  { event := event148147
    frameStart := 148123 },
  { event := event148148
    frameStart := 148123 },
  { event := event148149
    frameStart := 148123 },
  { event := event148150
    frameStart := 148123 },
  { event := event148151
    frameStart := 148123 },
  { event := event148152
    frameStart := 148123 },
  { event := event148153
    frameStart := 148123 },
  { event := event148154
    frameStart := 148123 },
  { event := event148155
    frameStart := 148123 },
  { event := event148156
    frameStart := 148123 },
  { event := event148157
    frameStart := 148123 },
  { event := event148158
    frameStart := 148123 },
  { event := event148159
    frameStart := 148123 }
]

def eventLeaf9260 : Array AnnotatedEvent := #[
  { event := event148160
    frameStart := 148123 },
  { event := event148161
    frameStart := 148123 },
  { event := event148162
    frameStart := 148123 },
  { event := event148163
    frameStart := 148123 },
  { event := event148164
    frameStart := 148123 },
  { event := event148165
    frameStart := 148123 },
  { event := event148166
    frameStart := 148123 },
  { event := event148167
    frameStart := 148123 },
  { event := event148168
    frameStart := 148123 },
  { event := event148169
    frameStart := 148123 },
  { event := event148170
    frameStart := 148123 },
  { event := event148171
    frameStart := 148123 },
  { event := event148172
    frameStart := 148123 },
  { event := event148173
    frameStart := 148123 },
  { event := event148174
    frameStart := 148123 },
  { event := event148175
    frameStart := 148123 }
]

def eventLeaf9261 : Array AnnotatedEvent := #[
  { event := event148176
    frameStart := 148123 },
  { event := event148177
    frameStart := 148177 },
  { event := event148178
    frameStart := 148177 },
  { event := event148179
    frameStart := 148177 },
  { event := event148180
    frameStart := 148177 },
  { event := event148181
    frameStart := 148177 },
  { event := event148182
    frameStart := 148177 },
  { event := event148183
    frameStart := 148177 },
  { event := event148184
    frameStart := 148177 },
  { event := event148185
    frameStart := 148177 },
  { event := event148186
    frameStart := 148177 },
  { event := event148187
    frameStart := 148177 },
  { event := event148188
    frameStart := 148177 },
  { event := event148189
    frameStart := 148177 },
  { event := event148190
    frameStart := 148177 },
  { event := event148191
    frameStart := 148177 }
]

def eventLeaf9262 : Array AnnotatedEvent := #[
  { event := event148192
    frameStart := 148177 },
  { event := event148193
    frameStart := 148177 },
  { event := event148194
    frameStart := 148177 },
  { event := event148195
    frameStart := 148177 },
  { event := event148196
    frameStart := 148177 },
  { event := event148197
    frameStart := 148177 },
  { event := event148198
    frameStart := 148177 },
  { event := event148199
    frameStart := 148177 },
  { event := event148200
    frameStart := 148177 },
  { event := event148201
    frameStart := 148177 },
  { event := event148202
    frameStart := 148177 },
  { event := event148203
    frameStart := 148177 },
  { event := event148204
    frameStart := 148177 },
  { event := event148205
    frameStart := 148177 },
  { event := event148206
    frameStart := 148177 },
  { event := event148207
    frameStart := 148177 }
]

def eventLeaf9263 : Array AnnotatedEvent := #[
  { event := event148208
    frameStart := 148177 },
  { event := event148209
    frameStart := 148177 },
  { event := event148210
    frameStart := 148177 },
  { event := event148211
    frameStart := 148177 },
  { event := event148212
    frameStart := 148177 },
  { event := event148213
    frameStart := 148177 },
  { event := event148214
    frameStart := 148177 },
  { event := event148215
    frameStart := 148177 },
  { event := event148216
    frameStart := 148177 },
  { event := event148217
    frameStart := 148177 },
  { event := event148218
    frameStart := 148177 },
  { event := event148219
    frameStart := 148177 },
  { event := event148220
    frameStart := 148177 },
  { event := event148221
    frameStart := 148177 },
  { event := event148222
    frameStart := 148177 },
  { event := event148223
    frameStart := 148177 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events578
