import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events570

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event145920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36449⟩⟩) 0 ⟨36081⟩ 145919

def event145921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36449⟩⟩) 1 ⟨36448⟩ 145896

def event145922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36449⟩⟩) (.product (.predecessor 0 145920 .coefficient) (.predecessor 1 145921 .coefficient) (⟨false, false, none, none, none⟩))

def event145923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36449⟩⟩, .operator (⟨145919, 0⟩, ⟨145896, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩)

def event145924 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36449⟩⟩, .operator (⟨145919, 1⟩, ⟨145896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩)

def event145925 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36449⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36448⟩⟩) ⟨35837⟩ 145893)

def event145926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36449⟩⟩, .relation 145925 0, ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (-1)⟩)

def exact145927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (-1)⟩]

theorem exact145927RawTermsValid :
    exact145927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36449⟩⟩) exact145927RawTerms .large 145922 .exactZero (none)

def event145928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34868⟩⟩) 0 ⟨34693⟩ 145885

def event145929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34868⟩⟩) (.authority (.programFamilyFact))

def exact145930RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩]

theorem exact145930RawTermsValid :
    exact145930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34868⟩⟩) exact145930RawTerms (.finite 40) 145929 .exactZero (none)

def event145931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34870⟩⟩) 0 ⟨6908⟩ 145907

def event145932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34870⟩⟩) 1 ⟨34868⟩ 145930

def event145933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34870⟩⟩) (.product (.predecessor 0 145931 .coefficient) (.predecessor 1 145932 .coefficient) (⟨false, true, none, none, some 1⟩))

def event145934 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34870⟩⟩, .operator (⟨145907, 0⟩, ⟨145930, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145935RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145935RawTermsValid :
    exact145935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34870⟩⟩) exact145935RawTerms .large 145933 .exactZero (none)

def event145936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7221⟩⟩) 0 ⟨7177⟩ 145889

def event145937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7221⟩⟩) (.authority (.operator))

def exact145938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩]

theorem exact145938RawTermsValid :
    exact145938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7221⟩⟩) exact145938RawTerms .large 145937 .exactZero (none)

def event145939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34871⟩⟩) 0 ⟨7221⟩ 145938

def event145940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34871⟩⟩) 1 ⟨34870⟩ 145935

def event145941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34871⟩⟩) (.sum [.predecessor 0 145939 .coefficient, .predecessor 1 145940 .coefficient])

def exact145942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145942RawTermsValid :
    exact145942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34871⟩⟩) exact145942RawTerms .large 145941 .exactZero (none)

def event145943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36453⟩⟩) 0 ⟨34871⟩ 145942

def event145944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36453⟩⟩) 1 ⟨36449⟩ 145927

def event145945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36453⟩⟩) (.sum [.predecessor 0 145943 .coefficient, .predecessor 1 145944 .coefficient])

def exact145946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145946RawTermsValid :
    exact145946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36453⟩⟩) exact145946RawTerms .large 145945 .exactZero (none)

def event145947 : Event := .preFoldPolynomial 145946 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact145948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event145948 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36453⟩⟩) 145947 exact145948RawTerms .large 145945 .exactZero (none)

def event145949 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34693⟩⟩) ⟨⟨100⟩, ⟨82⟩, ⟨135⟩⟩ ⟨145791, 145949⟩

def event145950 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35355⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (1) 0 2 (.universal 145949 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) (none) 145948)

def event145951 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35355⟩⟩, .relation 145950 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩)

def event145952 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35355⟩⟩, .relation 145950 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩)

def event145953 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35355⟩⟩, .relation 145950 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩)

def event145954 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35355⟩⟩, .relation 145950 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145955RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145955RawTermsValid :
    exact145955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145955 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35355⟩⟩) exact145955RawTerms .large 145787 (.finite 202072841853861888) (some (145789))

def event145956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36451⟩⟩) 0 ⟨35355⟩ 145955

def event145957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36451⟩⟩) 1 ⟨36450⟩ 145777

def event145958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36451⟩⟩) (.sum [.predecessor 0 145956 .coefficient, .predecessor 1 145957 .coefficient])

def event145959 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36451⟩⟩, .operator (⟨145955, 0⟩, ⟨145777, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩)

def event145960 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36451⟩⟩, .operator (⟨145955, 2⟩, ⟨145777, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (-1)⟩)

def event145961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36451⟩⟩) (.sum [.result 145955 .summary, .result 145777 .summary])

def exact145962RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145962RawTermsValid :
    exact145962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36451⟩⟩) exact145962RawTerms .large 145958 (.finite 32192539770951767057087530795008) (some (145961))

def event145963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36452⟩⟩) 0 ⟨36451⟩ 145962

def event145964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36452⟩⟩) 1 ⟨7164⟩ 15642

def event145965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36452⟩⟩) (.product (.predecessor 0 145963 .coefficient) (.predecessor 1 145964 .coefficient) (⟨false, false, none, none, none⟩))

def event145966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36452⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) [⟨.result 15638 .coefficient, false, none⟩])

def event145967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36452⟩⟩) (.product (.result 145962 .summary) (.transfer 145966) (⟨false, false, none, none, none⟩))

def event145968 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36452⟩⟩, .operator (⟨145962, 0⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event145969 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36452⟩⟩, .operator (⟨145962, 1⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event145970 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36452⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635)

def event145971 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36452⟩⟩, .relation 145970 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145972RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145972RawTermsValid :
    exact145972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36452⟩⟩) exact145972RawTerms .large 145965 (.finite 345664763728542925759002774434880600145920) (some (145967))

def event145973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30177⟩⟩) 0 ⟨7177⟩ 15500

def event145974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30177⟩⟩) 1 ⟨30176⟩ 137289

def event145975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30177⟩⟩) (.authority (.operator))

def exact145976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩]

theorem exact145976RawTermsValid :
    exact145976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30177⟩⟩) exact145976RawTerms .large 145975 .exactZero (none)

def event145977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30788⟩⟩) 0 ⟨30177⟩ 145976

def event145978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30788⟩⟩) (.authority (.operator))

def exact145979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩]

theorem exact145979RawTermsValid :
    exact145979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30788⟩⟩) exact145979RawTerms (.finite 8192) 145978 .exactZero (none)

def event145980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30790⟩⟩) 0 ⟨30524⟩ 137573

def event145981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30790⟩⟩) 1 ⟨30788⟩ 145979

def event145982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30790⟩⟩) (.product (.predecessor 0 145980 .coefficient) (.predecessor 1 145981 .coefficient) (⟨false, false, none, none, none⟩))

def event145983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30790⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) [⟨.result 145979 .coefficient, false, none⟩])

def event145984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30790⟩⟩) (.product (.result 137573 .summary) (.transfer 145983) (⟨false, false, none, none, none⟩))

def event145985 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30790⟩⟩, .operator (⟨137573, 0⟩, ⟨145979, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩)

def event145986 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30790⟩⟩, .operator (⟨137573, 1⟩, ⟨145979, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩)

def event145987 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30790⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30788⟩⟩) ⟨30177⟩ 145976)

def event145988 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30790⟩⟩, .relation 145987 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (-1)⟩)

def exact145989RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (-1)⟩]

theorem exact145989RawTermsValid :
    exact145989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30790⟩⟩) exact145989RawTerms .large 145982 (.finite 32192146870060190229763897425920) (some (145984))

def event145990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29692⟩⟩) 0 ⟨29033⟩ 6233

def event145991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29692⟩⟩) (.authority (.relationPreimageSource ⟨80⟩))

def exact145992RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩]

theorem exact145992RawTermsValid :
    exact145992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29692⟩⟩) exact145992RawTerms (.finite 5647228698) 145991 .exactZero (none)

def event145993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29694⟩⟩) 0 ⟨29692⟩ 145992

def event145994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29694⟩⟩) 1 ⟨2370⟩ 4

def event145995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29694⟩⟩) (.scale (.predecessor 0 145993 .coefficient) (.value (.predecessor 1 145994 .coefficient)))

def exact145996RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩]

theorem exact145996RawTermsValid :
    exact145996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29694⟩⟩) exact145996RawTerms (.finite 5647228698) 145995 .exactZero (none)

def event145997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29695⟩⟩) 0 ⟨5473⟩ 134495

def event145998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29695⟩⟩) 1 ⟨29694⟩ 145996

def event145999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29695⟩⟩) (.product (.predecessor 0 145997 .coefficient) (.predecessor 1 145998 .coefficient) (⟨false, false, none, none, none⟩))

def event146000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29695⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) [⟨.result 145992 .coefficient, false, none⟩])

def event146001 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29695⟩⟩) (.product (.result 134495 .summary) (.transfer 146000) (⟨false, false, none, none, none⟩))

def event146002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29695⟩⟩, .operator (⟨134495, 0⟩, ⟨145996, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩)

def event146003 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29693⟩⟩)

def event146004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event146005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event146006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event146007 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event146008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event146009 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event146010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event146011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event146012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 146011

def event146013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 146009

def event146014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 146012 .coefficient) (.value (.predecessor 1 146013 .coefficient)))

def event146015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event146016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 146015

def event146017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 146007

def event146018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 146016 .coefficient, .predecessor 1 146017 .coefficient])

def event146019 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event146020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 146019

def event146021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 146005

def event146022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 146021 .coefficient))

def event146023 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event146024 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28606⟩⟩) 0 ⟨5469⟩ 146023

def event146025 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28606⟩⟩) (.authority (.programFamilyFact))

def exact146026RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28606⟩⟩], []⟩, (1)⟩]

theorem exact146026RawTermsValid :
    exact146026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146026 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28606⟩⟩) exact146026RawTerms (.finite 36) 146025 .exactZero (none)

def event146027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13176⟩⟩) 0 ⟨5469⟩ 146023

def event146028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13176⟩⟩) (.authority (.programFamilyFact))

def exact146029RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩], []⟩, (1)⟩]

theorem exact146029RawTermsValid :
    exact146029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13176⟩⟩) exact146029RawTerms (.finite 36) 146028 .exactZero (none)

def event146030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28607⟩⟩) 0 ⟨13176⟩ 146029

def event146031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28607⟩⟩) 1 ⟨28606⟩ 146026

def event146032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28607⟩⟩) (.product (.predecessor 0 146030 .coefficient) (.predecessor 1 146031 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event146033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28607⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], []⟩) [⟨.result 146029 .coefficient, true, some 1⟩, ⟨.result 146026 .coefficient, true, some 1⟩])

def event146034 : Event := .survivorFold (1) 146033

def exact146035RawTerms : List Term := []

theorem exact146035RawTermsValid :
    exact146035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28607⟩⟩) exact146035RawTerms (.finite 1296) 146032 (.finite 1296) (some (146033))

def event146036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28608⟩⟩) 0 ⟨28607⟩ 146035

def event146037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28608⟩⟩) (.identity (.predecessor 0 146036 .coefficient))

def event146038 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28608⟩⟩) (.finite 1296)

def event146039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29032⟩⟩) 0 ⟨28608⟩ 146038

def event146040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29032⟩⟩) (.authority (.programFamilyFact))

def exact146041RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], []⟩, (1)⟩]

theorem exact146041RawTermsValid :
    exact146041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146041 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29032⟩⟩) exact146041RawTerms (.finite 36) 146040 .exactZero (none)

def event146042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29033⟩⟩) 0 ⟨29032⟩ 146041

def event146043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29033⟩⟩) (.identity (.predecessor 0 146042 .coefficient))

def event146044 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29033⟩⟩) (.finite 36)

def event146045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29692⟩⟩) 0 ⟨29033⟩ 146044

def event146046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29692⟩⟩) (.authority (.relationPreimageSource ⟨80⟩))

def exact146047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩]

theorem exact146047RawTermsValid :
    exact146047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29692⟩⟩) exact146047RawTerms (.finite 5647228698) 146046 .exactZero (none)

def event146048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact146049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact146049RawTermsValid :
    exact146049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact146049RawTerms .large 146048 .exactZero (none)

def event146050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29693⟩⟩) 0 ⟨35⟩ 146049

def event146051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29693⟩⟩) 1 ⟨29692⟩ 146047

def event146052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29693⟩⟩) (.product (.predecessor 0 146050 .coefficient) (.predecessor 1 146051 .coefficient) (⟨false, false, none, none, none⟩))

def event146053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29693⟩⟩, .operator (⟨146049, 0⟩, ⟨146047, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩)

def exact146054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩]

theorem exact146054RawTermsValid :
    exact146054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29693⟩⟩) exact146054RawTerms .large 146052 .exactZero (none)

def event146055 : Event := .preFoldPolynomial 146054 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩] .exactZero none

def exact146056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩, (1)⟩]

def event146056 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29693⟩⟩) 146055 exact146056RawTerms .large 146052 .exactZero (none)

def event146057 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨30793⟩⟩)

def event146058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event146059 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event146060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event146061 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event146062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event146063 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event146064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event146065 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event146066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 146065

def event146067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 146063

def event146068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 146066 .coefficient) (.value (.predecessor 1 146067 .coefficient)))

def event146069 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event146070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 146069

def event146071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 146061

def event146072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 146070 .coefficient, .predecessor 1 146071 .coefficient])

def event146073 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event146074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 146073

def event146075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 146059

def event146076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 146075 .coefficient))

def event146077 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event146078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28606⟩⟩) 0 ⟨5469⟩ 146077

def event146079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28606⟩⟩) (.authority (.programFamilyFact))

def exact146080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28606⟩⟩], []⟩, (1)⟩]

theorem exact146080RawTermsValid :
    exact146080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28606⟩⟩) exact146080RawTerms (.finite 36) 146079 .exactZero (none)

def event146081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13176⟩⟩) 0 ⟨5469⟩ 146077

def event146082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13176⟩⟩) (.authority (.programFamilyFact))

def exact146083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩], []⟩, (1)⟩]

theorem exact146083RawTermsValid :
    exact146083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13176⟩⟩) exact146083RawTerms (.finite 36) 146082 .exactZero (none)

def event146084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28607⟩⟩) 0 ⟨13176⟩ 146083

def event146085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28607⟩⟩) 1 ⟨28606⟩ 146080

def event146086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28607⟩⟩) (.product (.predecessor 0 146084 .coefficient) (.predecessor 1 146085 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event146087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28607⟩⟩, .operator (⟨146083, 0⟩, ⟨146080, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], []⟩, (1)⟩)

def exact146088RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], []⟩, (1)⟩]

theorem exact146088RawTermsValid :
    exact146088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28607⟩⟩) exact146088RawTerms (.finite 1296) 146086 .exactZero (none)

def event146089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28608⟩⟩) 0 ⟨28607⟩ 146088

def event146090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28608⟩⟩) (.identity (.predecessor 0 146089 .coefficient))

def event146091 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28608⟩⟩) (.finite 1296)

def event146092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29032⟩⟩) 0 ⟨28608⟩ 146091

def event146093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29032⟩⟩) (.authority (.programFamilyFact))

def exact146094RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], []⟩, (1)⟩]

theorem exact146094RawTermsValid :
    exact146094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146094 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29032⟩⟩) exact146094RawTerms (.finite 36) 146093 .exactZero (none)

def event146095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29033⟩⟩) 0 ⟨29032⟩ 146094

def event146096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29033⟩⟩) (.identity (.predecessor 0 146095 .coefficient))

def event146097 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29033⟩⟩) (.finite 36)

def event146098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30176⟩⟩) 0 ⟨29033⟩ 146097

def event146099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30176⟩⟩) (.authority (.programFamilyFact))

def event146100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30176⟩⟩) (.finite 3720)

def event146101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event146102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30177⟩⟩) 0 ⟨7177⟩ 146101

def event146103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30177⟩⟩) 1 ⟨30176⟩ 146100

def event146104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30177⟩⟩) (.authority (.operator))

def exact146105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩]

theorem exact146105RawTermsValid :
    exact146105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30177⟩⟩) exact146105RawTerms .large 146104 .exactZero (none)

def event146106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30788⟩⟩) 0 ⟨30177⟩ 146105

def event146107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30788⟩⟩) (.authority (.operator))

def exact146108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩]

theorem exact146108RawTermsValid :
    exact146108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30788⟩⟩) exact146108RawTerms (.finite 8192) 146107 .exactZero (none)

def event146109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event146110 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event146111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30418⟩⟩) 0 ⟨29033⟩ 146097

def event146112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30418⟩⟩) 1 ⟨136⟩ 146110

def event146113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30418⟩⟩) (.sum [.predecessor 0 146111 .coefficient, .predecessor 1 146112 .coefficient])

def event146114 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30418⟩⟩) (.finite 36)

def event146115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30419⟩⟩) 0 ⟨30418⟩ 146114

def event146116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30419⟩⟩) (.identity (.predecessor 0 146115 .coefficient))

def exact146117RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], []⟩, (1)⟩]

theorem exact146117RawTermsValid :
    exact146117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30419⟩⟩) exact146117RawTerms (.finite 36) 146116 .exactZero (none)

def event146118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact146119RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact146119RawTermsValid :
    exact146119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact146119RawTerms .large 146118 .exactZero (none)

def event146120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30420⟩⟩) 0 ⟨6908⟩ 146119

def event146121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30420⟩⟩) 1 ⟨30419⟩ 146117

def event146122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30420⟩⟩) (.product (.predecessor 0 146120 .coefficient) (.predecessor 1 146121 .coefficient) (⟨false, false, none, none, none⟩))

def event146123 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30420⟩⟩, .operator (⟨146119, 0⟩, ⟨146117, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact146124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact146124RawTermsValid :
    exact146124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30420⟩⟩) exact146124RawTerms .large 146122 .exactZero (none)

def event146125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 146101

def event146126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact146127RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact146127RawTermsValid :
    exact146127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146127 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact146127RawTerms .large 146126 .exactZero (none)

def event146128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30421⟩⟩) 0 ⟨7190⟩ 146127

def event146129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30421⟩⟩) 1 ⟨30420⟩ 146124

def event146130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30421⟩⟩) (.sum [.predecessor 0 146128 .coefficient, .predecessor 1 146129 .coefficient])

def exact146131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact146131RawTermsValid :
    exact146131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30421⟩⟩) exact146131RawTerms .large 146130 .exactZero (none)

def event146132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30789⟩⟩) 0 ⟨30421⟩ 146131

def event146133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30789⟩⟩) 1 ⟨30788⟩ 146108

def event146134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30789⟩⟩) (.product (.predecessor 0 146132 .coefficient) (.predecessor 1 146133 .coefficient) (⟨false, false, none, none, none⟩))

def event146135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30789⟩⟩, .operator (⟨146131, 0⟩, ⟨146108, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩)

def event146136 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30789⟩⟩, .operator (⟨146131, 1⟩, ⟨146108, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩)

def event146137 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30789⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30788⟩⟩) ⟨30177⟩ 146105)

def event146138 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30789⟩⟩, .relation 146137 0, ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (-1)⟩)

def exact146139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (-1)⟩]

theorem exact146139RawTermsValid :
    exact146139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146139 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30789⟩⟩) exact146139RawTerms .large 146134 .exactZero (none)

def event146140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29211⟩⟩) 0 ⟨29033⟩ 146097

def event146141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29211⟩⟩) (.authority (.programFamilyFact))

def exact146142RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩]

theorem exact146142RawTermsValid :
    exact146142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146142 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29211⟩⟩) exact146142RawTerms (.finite 36) 146141 .exactZero (none)

def event146143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29213⟩⟩) 0 ⟨6908⟩ 146119

def event146144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29213⟩⟩) 1 ⟨29211⟩ 146142

def event146145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29213⟩⟩) (.product (.predecessor 0 146143 .coefficient) (.predecessor 1 146144 .coefficient) (⟨false, true, none, none, some 1⟩))

def event146146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29213⟩⟩, .operator (⟨146119, 0⟩, ⟨146142, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact146147RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact146147RawTermsValid :
    exact146147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29213⟩⟩) exact146147RawTerms .large 146145 .exactZero (none)

def event146148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7219⟩⟩) 0 ⟨7177⟩ 146101

def event146149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7219⟩⟩) (.authority (.operator))

def exact146150RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩]

theorem exact146150RawTermsValid :
    exact146150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7219⟩⟩) exact146150RawTerms .large 146149 .exactZero (none)

def event146151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29214⟩⟩) 0 ⟨7219⟩ 146150

def event146152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29214⟩⟩) 1 ⟨29213⟩ 146147

def event146153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29214⟩⟩) (.sum [.predecessor 0 146151 .coefficient, .predecessor 1 146152 .coefficient])

def exact146154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact146154RawTermsValid :
    exact146154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29214⟩⟩) exact146154RawTerms .large 146153 .exactZero (none)

def event146155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30793⟩⟩) 0 ⟨29214⟩ 146154

def event146156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30793⟩⟩) 1 ⟨30789⟩ 146139

def event146157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30793⟩⟩) (.sum [.predecessor 0 146155 .coefficient, .predecessor 1 146156 .coefficient])

def exact146158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact146158RawTermsValid :
    exact146158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30793⟩⟩) exact146158RawTerms .large 146157 .exactZero (none)

def event146159 : Event := .preFoldPolynomial 146158 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact146160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event146160 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨30793⟩⟩) 146159 exact146160RawTerms .large 146157 .exactZero (none)

def event146161 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨29033⟩⟩) ⟨⟨98⟩, ⟨80⟩, ⟨135⟩⟩ ⟨146003, 146161⟩

def event146162 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29695⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (1) 0 2 (.universal 146161 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (none) 146160)

def event146163 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29695⟩⟩, .relation 146162 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩)

def event146164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29695⟩⟩, .relation 146162 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩)

def event146165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29695⟩⟩, .relation 146162 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩)

def event146166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29695⟩⟩, .relation 146162 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact146167RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact146167RawTermsValid :
    exact146167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146167 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29695⟩⟩) exact146167RawTerms .large 145999 (.finite 202072841853861888) (some (146001))

def event146168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30791⟩⟩) 0 ⟨29695⟩ 146167

def event146169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30791⟩⟩) 1 ⟨30790⟩ 145989

def event146170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30791⟩⟩) (.sum [.predecessor 0 146168 .coefficient, .predecessor 1 146169 .coefficient])

def event146171 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30791⟩⟩, .operator (⟨146167, 0⟩, ⟨145989, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩, (1)⟩)

def event146172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30791⟩⟩, .operator (⟨146167, 2⟩, ⟨145989, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩, (-1)⟩)

def event146173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30791⟩⟩) (.sum [.result 146167 .summary, .result 145989 .summary])

def exact146174RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact146174RawTermsValid :
    exact146174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event146174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30791⟩⟩) exact146174RawTerms .large 146170 (.finite 32192146870060392302605751287808) (some (146173))

def event146175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30792⟩⟩) 0 ⟨30791⟩ 146174

def eventLeaf9120 : Array AnnotatedEvent := #[
  { event := event145920
    frameStart := 145845 },
  { event := event145921
    frameStart := 145845 },
  { event := event145922
    frameStart := 145845 },
  { event := event145923
    frameStart := 145845 },
  { event := event145924
    frameStart := 145845 },
  { event := event145925
    frameStart := 145845 },
  { event := event145926
    frameStart := 145845 },
  { event := event145927
    frameStart := 145845 },
  { event := event145928
    frameStart := 145845 },
  { event := event145929
    frameStart := 145845 },
  { event := event145930
    frameStart := 145845 },
  { event := event145931
    frameStart := 145845 },
  { event := event145932
    frameStart := 145845 },
  { event := event145933
    frameStart := 145845 },
  { event := event145934
    frameStart := 145845 },
  { event := event145935
    frameStart := 145845 }
]

def eventLeaf9121 : Array AnnotatedEvent := #[
  { event := event145936
    frameStart := 145845 },
  { event := event145937
    frameStart := 145845 },
  { event := event145938
    frameStart := 145845 },
  { event := event145939
    frameStart := 145845 },
  { event := event145940
    frameStart := 145845 },
  { event := event145941
    frameStart := 145845 },
  { event := event145942
    frameStart := 145845 },
  { event := event145943
    frameStart := 145845 },
  { event := event145944
    frameStart := 145845 },
  { event := event145945
    frameStart := 145845 },
  { event := event145946
    frameStart := 145845 },
  { event := event145947
    frameStart := 145845 },
  { event := event145948
    frameStart := 145845 },
  { event := event145949
    frameStart := 0 },
  { event := event145950
    frameStart := 0 },
  { event := event145951
    frameStart := 0 }
]

def eventLeaf9122 : Array AnnotatedEvent := #[
  { event := event145952
    frameStart := 0 },
  { event := event145953
    frameStart := 0 },
  { event := event145954
    frameStart := 0 },
  { event := event145955
    frameStart := 0 },
  { event := event145956
    frameStart := 0 },
  { event := event145957
    frameStart := 0 },
  { event := event145958
    frameStart := 0 },
  { event := event145959
    frameStart := 0 },
  { event := event145960
    frameStart := 0 },
  { event := event145961
    frameStart := 0 },
  { event := event145962
    frameStart := 0 },
  { event := event145963
    frameStart := 0 },
  { event := event145964
    frameStart := 0 },
  { event := event145965
    frameStart := 0 },
  { event := event145966
    frameStart := 0 },
  { event := event145967
    frameStart := 0 }
]

def eventLeaf9123 : Array AnnotatedEvent := #[
  { event := event145968
    frameStart := 0 },
  { event := event145969
    frameStart := 0 },
  { event := event145970
    frameStart := 0 },
  { event := event145971
    frameStart := 0 },
  { event := event145972
    frameStart := 0 },
  { event := event145973
    frameStart := 0 },
  { event := event145974
    frameStart := 0 },
  { event := event145975
    frameStart := 0 },
  { event := event145976
    frameStart := 0 },
  { event := event145977
    frameStart := 0 },
  { event := event145978
    frameStart := 0 },
  { event := event145979
    frameStart := 0 },
  { event := event145980
    frameStart := 0 },
  { event := event145981
    frameStart := 0 },
  { event := event145982
    frameStart := 0 },
  { event := event145983
    frameStart := 0 }
]

def eventLeaf9124 : Array AnnotatedEvent := #[
  { event := event145984
    frameStart := 0 },
  { event := event145985
    frameStart := 0 },
  { event := event145986
    frameStart := 0 },
  { event := event145987
    frameStart := 0 },
  { event := event145988
    frameStart := 0 },
  { event := event145989
    frameStart := 0 },
  { event := event145990
    frameStart := 0 },
  { event := event145991
    frameStart := 0 },
  { event := event145992
    frameStart := 0 },
  { event := event145993
    frameStart := 0 },
  { event := event145994
    frameStart := 0 },
  { event := event145995
    frameStart := 0 },
  { event := event145996
    frameStart := 0 },
  { event := event145997
    frameStart := 0 },
  { event := event145998
    frameStart := 0 },
  { event := event145999
    frameStart := 0 }
]

def eventLeaf9125 : Array AnnotatedEvent := #[
  { event := event146000
    frameStart := 0 },
  { event := event146001
    frameStart := 0 },
  { event := event146002
    frameStart := 0 },
  { event := event146003
    frameStart := 146003 },
  { event := event146004
    frameStart := 146003 },
  { event := event146005
    frameStart := 146003 },
  { event := event146006
    frameStart := 146003 },
  { event := event146007
    frameStart := 146003 },
  { event := event146008
    frameStart := 146003 },
  { event := event146009
    frameStart := 146003 },
  { event := event146010
    frameStart := 146003 },
  { event := event146011
    frameStart := 146003 },
  { event := event146012
    frameStart := 146003 },
  { event := event146013
    frameStart := 146003 },
  { event := event146014
    frameStart := 146003 },
  { event := event146015
    frameStart := 146003 }
]

def eventLeaf9126 : Array AnnotatedEvent := #[
  { event := event146016
    frameStart := 146003 },
  { event := event146017
    frameStart := 146003 },
  { event := event146018
    frameStart := 146003 },
  { event := event146019
    frameStart := 146003 },
  { event := event146020
    frameStart := 146003 },
  { event := event146021
    frameStart := 146003 },
  { event := event146022
    frameStart := 146003 },
  { event := event146023
    frameStart := 146003 },
  { event := event146024
    frameStart := 146003 },
  { event := event146025
    frameStart := 146003 },
  { event := event146026
    frameStart := 146003 },
  { event := event146027
    frameStart := 146003 },
  { event := event146028
    frameStart := 146003 },
  { event := event146029
    frameStart := 146003 },
  { event := event146030
    frameStart := 146003 },
  { event := event146031
    frameStart := 146003 }
]

def eventLeaf9127 : Array AnnotatedEvent := #[
  { event := event146032
    frameStart := 146003 },
  { event := event146033
    frameStart := 146003 },
  { event := event146034
    frameStart := 146003 },
  { event := event146035
    frameStart := 146003 },
  { event := event146036
    frameStart := 146003 },
  { event := event146037
    frameStart := 146003 },
  { event := event146038
    frameStart := 146003 },
  { event := event146039
    frameStart := 146003 },
  { event := event146040
    frameStart := 146003 },
  { event := event146041
    frameStart := 146003 },
  { event := event146042
    frameStart := 146003 },
  { event := event146043
    frameStart := 146003 },
  { event := event146044
    frameStart := 146003 },
  { event := event146045
    frameStart := 146003 },
  { event := event146046
    frameStart := 146003 },
  { event := event146047
    frameStart := 146003 }
]

def eventLeaf9128 : Array AnnotatedEvent := #[
  { event := event146048
    frameStart := 146003 },
  { event := event146049
    frameStart := 146003 },
  { event := event146050
    frameStart := 146003 },
  { event := event146051
    frameStart := 146003 },
  { event := event146052
    frameStart := 146003 },
  { event := event146053
    frameStart := 146003 },
  { event := event146054
    frameStart := 146003 },
  { event := event146055
    frameStart := 146003 },
  { event := event146056
    frameStart := 146003 },
  { event := event146057
    frameStart := 146057 },
  { event := event146058
    frameStart := 146057 },
  { event := event146059
    frameStart := 146057 },
  { event := event146060
    frameStart := 146057 },
  { event := event146061
    frameStart := 146057 },
  { event := event146062
    frameStart := 146057 },
  { event := event146063
    frameStart := 146057 }
]

def eventLeaf9129 : Array AnnotatedEvent := #[
  { event := event146064
    frameStart := 146057 },
  { event := event146065
    frameStart := 146057 },
  { event := event146066
    frameStart := 146057 },
  { event := event146067
    frameStart := 146057 },
  { event := event146068
    frameStart := 146057 },
  { event := event146069
    frameStart := 146057 },
  { event := event146070
    frameStart := 146057 },
  { event := event146071
    frameStart := 146057 },
  { event := event146072
    frameStart := 146057 },
  { event := event146073
    frameStart := 146057 },
  { event := event146074
    frameStart := 146057 },
  { event := event146075
    frameStart := 146057 },
  { event := event146076
    frameStart := 146057 },
  { event := event146077
    frameStart := 146057 },
  { event := event146078
    frameStart := 146057 },
  { event := event146079
    frameStart := 146057 }
]

def eventLeaf9130 : Array AnnotatedEvent := #[
  { event := event146080
    frameStart := 146057 },
  { event := event146081
    frameStart := 146057 },
  { event := event146082
    frameStart := 146057 },
  { event := event146083
    frameStart := 146057 },
  { event := event146084
    frameStart := 146057 },
  { event := event146085
    frameStart := 146057 },
  { event := event146086
    frameStart := 146057 },
  { event := event146087
    frameStart := 146057 },
  { event := event146088
    frameStart := 146057 },
  { event := event146089
    frameStart := 146057 },
  { event := event146090
    frameStart := 146057 },
  { event := event146091
    frameStart := 146057 },
  { event := event146092
    frameStart := 146057 },
  { event := event146093
    frameStart := 146057 },
  { event := event146094
    frameStart := 146057 },
  { event := event146095
    frameStart := 146057 }
]

def eventLeaf9131 : Array AnnotatedEvent := #[
  { event := event146096
    frameStart := 146057 },
  { event := event146097
    frameStart := 146057 },
  { event := event146098
    frameStart := 146057 },
  { event := event146099
    frameStart := 146057 },
  { event := event146100
    frameStart := 146057 },
  { event := event146101
    frameStart := 146057 },
  { event := event146102
    frameStart := 146057 },
  { event := event146103
    frameStart := 146057 },
  { event := event146104
    frameStart := 146057 },
  { event := event146105
    frameStart := 146057 },
  { event := event146106
    frameStart := 146057 },
  { event := event146107
    frameStart := 146057 },
  { event := event146108
    frameStart := 146057 },
  { event := event146109
    frameStart := 146057 },
  { event := event146110
    frameStart := 146057 },
  { event := event146111
    frameStart := 146057 }
]

def eventLeaf9132 : Array AnnotatedEvent := #[
  { event := event146112
    frameStart := 146057 },
  { event := event146113
    frameStart := 146057 },
  { event := event146114
    frameStart := 146057 },
  { event := event146115
    frameStart := 146057 },
  { event := event146116
    frameStart := 146057 },
  { event := event146117
    frameStart := 146057 },
  { event := event146118
    frameStart := 146057 },
  { event := event146119
    frameStart := 146057 },
  { event := event146120
    frameStart := 146057 },
  { event := event146121
    frameStart := 146057 },
  { event := event146122
    frameStart := 146057 },
  { event := event146123
    frameStart := 146057 },
  { event := event146124
    frameStart := 146057 },
  { event := event146125
    frameStart := 146057 },
  { event := event146126
    frameStart := 146057 },
  { event := event146127
    frameStart := 146057 }
]

def eventLeaf9133 : Array AnnotatedEvent := #[
  { event := event146128
    frameStart := 146057 },
  { event := event146129
    frameStart := 146057 },
  { event := event146130
    frameStart := 146057 },
  { event := event146131
    frameStart := 146057 },
  { event := event146132
    frameStart := 146057 },
  { event := event146133
    frameStart := 146057 },
  { event := event146134
    frameStart := 146057 },
  { event := event146135
    frameStart := 146057 },
  { event := event146136
    frameStart := 146057 },
  { event := event146137
    frameStart := 146057 },
  { event := event146138
    frameStart := 146057 },
  { event := event146139
    frameStart := 146057 },
  { event := event146140
    frameStart := 146057 },
  { event := event146141
    frameStart := 146057 },
  { event := event146142
    frameStart := 146057 },
  { event := event146143
    frameStart := 146057 }
]

def eventLeaf9134 : Array AnnotatedEvent := #[
  { event := event146144
    frameStart := 146057 },
  { event := event146145
    frameStart := 146057 },
  { event := event146146
    frameStart := 146057 },
  { event := event146147
    frameStart := 146057 },
  { event := event146148
    frameStart := 146057 },
  { event := event146149
    frameStart := 146057 },
  { event := event146150
    frameStart := 146057 },
  { event := event146151
    frameStart := 146057 },
  { event := event146152
    frameStart := 146057 },
  { event := event146153
    frameStart := 146057 },
  { event := event146154
    frameStart := 146057 },
  { event := event146155
    frameStart := 146057 },
  { event := event146156
    frameStart := 146057 },
  { event := event146157
    frameStart := 146057 },
  { event := event146158
    frameStart := 146057 },
  { event := event146159
    frameStart := 146057 }
]

def eventLeaf9135 : Array AnnotatedEvent := #[
  { event := event146160
    frameStart := 146057 },
  { event := event146161
    frameStart := 0 },
  { event := event146162
    frameStart := 0 },
  { event := event146163
    frameStart := 0 },
  { event := event146164
    frameStart := 0 },
  { event := event146165
    frameStart := 0 },
  { event := event146166
    frameStart := 0 },
  { event := event146167
    frameStart := 0 },
  { event := event146168
    frameStart := 0 },
  { event := event146169
    frameStart := 0 },
  { event := event146170
    frameStart := 0 },
  { event := event146171
    frameStart := 0 },
  { event := event146172
    frameStart := 0 },
  { event := event146173
    frameStart := 0 },
  { event := event146174
    frameStart := 0 },
  { event := event146175
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events570
