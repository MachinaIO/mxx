import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events023

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event5888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66309⟩⟩) 0 ⟨66308⟩ 5887

def event5889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66309⟩⟩) 1 ⟨6870⟩ 623

def event5890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66309⟩⟩) (.product (.predecessor 0 5888 .coefficient) (.predecessor 1 5889 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨66309⟩⟩, .operator (⟨5887, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩)

def exact5892RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact5892RawTermsValid :
    exact5892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66309⟩⟩) exact5892RawTerms (.finite 226487908831958288795280) 5890 .exactZero (none)

def event5893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63009⟩⟩) 0 ⟨62777⟩ 5554

def event5894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63009⟩⟩) (.authority (.programFamilyFact))

def exact5895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩]

theorem exact5895RawTermsValid :
    exact5895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63009⟩⟩) exact5895RawTerms (.finite 22) 5894 .exactZero (none)

def event5896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63010⟩⟩) 0 ⟨63009⟩ 5895

def event5897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63010⟩⟩) 1 ⟨6732⟩ 633

def event5898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63010⟩⟩) (.product (.predecessor 0 5896 .coefficient) (.predecessor 1 5897 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63010⟩⟩, .operator (⟨5895, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩)

def exact5900RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩]

theorem exact5900RawTermsValid :
    exact5900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63010⟩⟩) exact5900RawTerms (.finite 224377773035387248837560) 5898 .exactZero (none)

def event5901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60029⟩⟩) 0 ⟨59797⟩ 5577

def event5902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60029⟩⟩) (.authority (.programFamilyFact))

def exact5903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩]

theorem exact5903RawTermsValid :
    exact5903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60029⟩⟩) exact5903RawTerms (.finite 18) 5902 .exactZero (none)

def event5904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60030⟩⟩) 0 ⟨60029⟩ 5903

def event5905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60030⟩⟩) 1 ⟨6736⟩ 643

def event5906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60030⟩⟩) (.product (.predecessor 0 5904 .coefficient) (.predecessor 1 5905 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60030⟩⟩, .operator (⟨5903, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩)

def exact5908RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩]

theorem exact5908RawTermsValid :
    exact5908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60030⟩⟩) exact5908RawTerms (.finite 222230617312560576599880) 5906 .exactZero (none)

def event5909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57049⟩⟩) 0 ⟨56817⟩ 5600

def event5910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57049⟩⟩) (.authority (.programFamilyFact))

def exact5911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩]

theorem exact5911RawTermsValid :
    exact5911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57049⟩⟩) exact5911RawTerms (.finite 16) 5910 .exactZero (none)

def event5912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57050⟩⟩) 0 ⟨57049⟩ 5911

def event5913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57050⟩⟩) 1 ⟨6741⟩ 653

def event5914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57050⟩⟩) (.product (.predecessor 0 5912 .coefficient) (.predecessor 1 5913 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57050⟩⟩, .operator (⟨5911, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩)

def exact5916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩]

theorem exact5916RawTermsValid :
    exact5916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57050⟩⟩) exact5916RawTerms (.finite 220778129617707239497920) 5914 .exactZero (none)

def event5917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54069⟩⟩) 0 ⟨53837⟩ 5623

def event5918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54069⟩⟩) (.authority (.programFamilyFact))

def exact5919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩]

theorem exact5919RawTermsValid :
    exact5919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54069⟩⟩) exact5919RawTerms (.finite 12) 5918 .exactZero (none)

def event5920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54070⟩⟩) 0 ⟨54069⟩ 5919

def event5921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54070⟩⟩) 1 ⟨6757⟩ 663

def event5922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54070⟩⟩) (.product (.predecessor 0 5920 .coefficient) (.predecessor 1 5921 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54070⟩⟩, .operator (⟨5919, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩)

def exact5924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩]

theorem exact5924RawTermsValid :
    exact5924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54070⟩⟩) exact5924RawTerms (.finite 216532396355828254122960) 5922 .exactZero (none)

def event5925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51089⟩⟩) 0 ⟨50857⟩ 5646

def event5926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51089⟩⟩) (.authority (.programFamilyFact))

def exact5927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩]

theorem exact5927RawTermsValid :
    exact5927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51089⟩⟩) exact5927RawTerms (.finite 10) 5926 .exactZero (none)

def event5928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51090⟩⟩) 0 ⟨51089⟩ 5927

def event5929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51090⟩⟩) 1 ⟨6768⟩ 673

def event5930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51090⟩⟩) (.product (.predecessor 0 5928 .coefficient) (.predecessor 1 5929 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5931 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51090⟩⟩, .operator (⟨5927, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩)

def exact5932RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩]

theorem exact5932RawTermsValid :
    exact5932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51090⟩⟩) exact5932RawTerms (.finite 213251602471649038151400) 5930 .exactZero (none)

def event5933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32025⟩⟩) 0 ⟨31797⟩ 5669

def event5934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32025⟩⟩) (.authority (.programFamilyFact))

def exact5935RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩]

theorem exact5935RawTermsValid :
    exact5935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32025⟩⟩) exact5935RawTerms (.finite 6) 5934 .exactZero (none)

def event5936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32026⟩⟩) 0 ⟨32025⟩ 5935

def event5937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32026⟩⟩) 1 ⟨6794⟩ 683

def event5938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32026⟩⟩) (.product (.predecessor 0 5936 .coefficient) (.predecessor 1 5937 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5939 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32026⟩⟩, .operator (⟨5935, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩)

def exact5940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩]

theorem exact5940RawTermsValid :
    exact5940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32026⟩⟩) exact5940RawTerms (.finite 201065796616126235971320) 5938 .exactZero (none)

def event5941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22005⟩⟩) 0 ⟨21777⟩ 5692

def event5942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22005⟩⟩) (.authority (.programFamilyFact))

def exact5943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩]

theorem exact5943RawTermsValid :
    exact5943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22005⟩⟩) exact5943RawTerms (.finite 4) 5942 .exactZero (none)

def event5944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22006⟩⟩) 0 ⟨22005⟩ 5943

def event5945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22006⟩⟩) 1 ⟨6822⟩ 693

def event5946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22006⟩⟩) (.product (.predecessor 0 5944 .coefficient) (.predecessor 1 5945 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5947 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22006⟩⟩, .operator (⟨5943, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩)

def exact5948RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩]

theorem exact5948RawTermsValid :
    exact5948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22006⟩⟩) exact5948RawTerms (.finite 187661410175051153573232) 5946 .exactZero (none)

def event5949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18785⟩⟩) 0 ⟨18557⟩ 5715

def event5950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18785⟩⟩) (.authority (.programFamilyFact))

def exact5951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩]

theorem exact5951RawTermsValid :
    exact5951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18785⟩⟩) exact5951RawTerms (.finite 3) 5950 .exactZero (none)

def event5952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18786⟩⟩) 0 ⟨18785⟩ 5951

def event5953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18786⟩⟩) 1 ⟨6846⟩ 703

def event5954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18786⟩⟩) (.product (.predecessor 0 5952 .coefficient) (.predecessor 1 5953 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18786⟩⟩, .operator (⟨5951, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩)

def exact5956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩]

theorem exact5956RawTermsValid :
    exact5956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18786⟩⟩) exact5956RawTerms (.finite 175932572039110456474905) 5954 .exactZero (none)

def event5957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15966⟩⟩) 0 ⟨15757⟩ 5738

def event5958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15966⟩⟩) (.authority (.programFamilyFact))

def exact5959RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5959RawTermsValid :
    exact5959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15966⟩⟩) exact5959RawTerms (.finite 2) 5958 .exactZero (none)

def event5960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15967⟩⟩) 0 ⟨15966⟩ 5959

def event5961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15967⟩⟩) 1 ⟨6863⟩ 713

def event5962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15967⟩⟩) (.product (.predecessor 0 5960 .coefficient) (.predecessor 1 5961 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event5963 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15967⟩⟩, .operator (⟨5959, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩)

def exact5964RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5964RawTermsValid :
    exact5964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15967⟩⟩) exact5964RawTerms (.finite 156384508479209294644360) 5962 .exactZero (none)

def event5965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15968⟩⟩) 0 ⟨6728⟩ 728

def event5966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15968⟩⟩) 1 ⟨15967⟩ 5964

def event5967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15968⟩⟩) (.sum [.predecessor 0 5965 .coefficient, .predecessor 1 5966 .coefficient])

def exact5968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5968RawTermsValid :
    exact5968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15968⟩⟩) exact5968RawTerms (.finite 156384508479209294644360) 5967 .exactZero (none)

def event5969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18787⟩⟩) 0 ⟨15968⟩ 5968

def event5970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18787⟩⟩) 1 ⟨18786⟩ 5956

def event5971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18787⟩⟩) (.sum [.predecessor 0 5969 .coefficient, .predecessor 1 5970 .coefficient])

def exact5972RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5972RawTermsValid :
    exact5972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18787⟩⟩) exact5972RawTerms (.finite 332317080518319751119265) 5971 .exactZero (none)

def event5973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22007⟩⟩) 0 ⟨18787⟩ 5972

def event5974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22007⟩⟩) 1 ⟨22006⟩ 5948

def event5975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22007⟩⟩) (.sum [.predecessor 0 5973 .coefficient, .predecessor 1 5974 .coefficient])

def exact5976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5976RawTermsValid :
    exact5976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22007⟩⟩) exact5976RawTerms (.finite 519978490693370904692497) 5975 .exactZero (none)

def event5977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32027⟩⟩) 0 ⟨22007⟩ 5976

def event5978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32027⟩⟩) 1 ⟨32026⟩ 5940

def event5979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32027⟩⟩) (.sum [.predecessor 0 5977 .coefficient, .predecessor 1 5978 .coefficient])

def exact5980RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5980RawTermsValid :
    exact5980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32027⟩⟩) exact5980RawTerms (.finite 721044287309497140663817) 5979 .exactZero (none)

def event5981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51091⟩⟩) 0 ⟨32027⟩ 5980

def event5982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51091⟩⟩) 1 ⟨51090⟩ 5932

def event5983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51091⟩⟩) (.sum [.predecessor 0 5981 .coefficient, .predecessor 1 5982 .coefficient])

def exact5984RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5984RawTermsValid :
    exact5984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51091⟩⟩) exact5984RawTerms (.finite 934295889781146178815217) 5983 .exactZero (none)

def event5985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54071⟩⟩) 0 ⟨51091⟩ 5984

def event5986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54071⟩⟩) 1 ⟨54070⟩ 5924

def event5987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54071⟩⟩) (.sum [.predecessor 0 5985 .coefficient, .predecessor 1 5986 .coefficient])

def exact5988RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5988RawTermsValid :
    exact5988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5988 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54071⟩⟩) exact5988RawTerms (.finite 1150828286136974432938177) 5987 .exactZero (none)

def event5989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57051⟩⟩) 0 ⟨54071⟩ 5988

def event5990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57051⟩⟩) 1 ⟨57050⟩ 5916

def event5991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57051⟩⟩) (.sum [.predecessor 0 5989 .coefficient, .predecessor 1 5990 .coefficient])

def exact5992RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5992RawTermsValid :
    exact5992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57051⟩⟩) exact5992RawTerms (.finite 1371606415754681672436097) 5991 .exactZero (none)

def event5993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60031⟩⟩) 0 ⟨57051⟩ 5992

def event5994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60031⟩⟩) 1 ⟨60030⟩ 5908

def event5995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60031⟩⟩) (.sum [.predecessor 0 5993 .coefficient, .predecessor 1 5994 .coefficient])

def exact5996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact5996RawTermsValid :
    exact5996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60031⟩⟩) exact5996RawTerms (.finite 1593837033067242249035977) 5995 .exactZero (none)

def event5997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63011⟩⟩) 0 ⟨60031⟩ 5996

def event5998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63011⟩⟩) 1 ⟨63010⟩ 5900

def event5999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63011⟩⟩) (.sum [.predecessor 0 5997 .coefficient, .predecessor 1 5998 .coefficient])

def exact6000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩]

theorem exact6000RawTermsValid :
    exact6000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63011⟩⟩) exact6000RawTerms (.finite 1818214806102629497873537) 5999 .exactZero (none)

def event6001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66310⟩⟩) 0 ⟨63011⟩ 6000

def event6002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66310⟩⟩) 1 ⟨66309⟩ 5892

def event6003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66310⟩⟩) (.sum [.predecessor 0 6001 .coefficient, .predecessor 1 6002 .coefficient])

def exact6004RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6004RawTermsValid :
    exact6004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66310⟩⟩) exact6004RawTerms (.finite 2044702714934587786668817) 6003 .exactZero (none)

def event6005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66311⟩⟩) 0 ⟨66310⟩ 6004

def event6006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66311⟩⟩) 1 ⟨26571⟩ 5884

def event6007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66311⟩⟩) (.sum [.predecessor 0 6005 .coefficient, .predecessor 1 6006 .coefficient])

def exact6008RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6008RawTermsValid :
    exact6008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6008 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66311⟩⟩) exact6008RawTerms (.finite 2271712485307633536959017) 6007 .exactZero (none)

def event6009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66312⟩⟩) 0 ⟨66311⟩ 6008

def event6010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66312⟩⟩) 1 ⟨29251⟩ 5876

def event6011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66312⟩⟩) (.sum [.predecessor 0 6009 .coefficient, .predecessor 1 6010 .coefficient])

def exact6012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6012RawTermsValid :
    exact6012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66312⟩⟩) exact6012RawTerms (.finite 2499949335520533588602137) 6011 .exactZero (none)

def event6013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66313⟩⟩) 0 ⟨66312⟩ 6012

def event6014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66313⟩⟩) 1 ⟨34908⟩ 5868

def event6015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66313⟩⟩) (.sum [.predecessor 0 6013 .coefficient, .predecessor 1 6014 .coefficient])

def exact6016RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6016RawTermsValid :
    exact6016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66313⟩⟩) exact6016RawTerms (.finite 2728804713782791092959737) 6015 .exactZero (none)

def event6017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66314⟩⟩) 0 ⟨66313⟩ 6016

def event6018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66314⟩⟩) 1 ⟨37588⟩ 5860

def event6019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66314⟩⟩) (.sum [.predecessor 0 6017 .coefficient, .predecessor 1 6018 .coefficient])

def exact6020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6020RawTermsValid :
    exact6020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66314⟩⟩) exact6020RawTerms (.finite 2957926202950004710694497) 6019 .exactZero (none)

def event6021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66315⟩⟩) 0 ⟨66314⟩ 6020

def event6022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66315⟩⟩) 1 ⟨40271⟩ 5852

def event6023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66315⟩⟩) (.sum [.predecessor 0 6021 .coefficient, .predecessor 1 6022 .coefficient])

def exact6024RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6024RawTermsValid :
    exact6024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66315⟩⟩) exact6024RawTerms (.finite 3187511970717354526236217) 6023 .exactZero (none)

def event6025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66316⟩⟩) 0 ⟨66315⟩ 6024

def event6026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66316⟩⟩) 1 ⟨42951⟩ 5844

def event6027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66316⟩⟩) (.sum [.predecessor 0 6025 .coefficient, .predecessor 1 6026 .coefficient])

def exact6028RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6028RawTermsValid :
    exact6028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66316⟩⟩) exact6028RawTerms (.finite 3417662756781096507033577) 6027 .exactZero (none)

def event6029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66317⟩⟩) 0 ⟨66316⟩ 6028

def event6030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66317⟩⟩) 1 ⟨45628⟩ 5836

def event6031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66317⟩⟩) (.sum [.predecessor 0 6029 .coefficient, .predecessor 1 6030 .coefficient])

def exact6032RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6032RawTermsValid :
    exact6032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66317⟩⟩) exact6032RawTerms (.finite 3648263642165693263543057) 6031 .exactZero (none)

def event6033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66318⟩⟩) 0 ⟨66317⟩ 6032

def event6034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66318⟩⟩) 1 ⟨48308⟩ 5828

def event6035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66318⟩⟩) (.sum [.predecessor 0 6033 .coefficient, .predecessor 1 6034 .coefficient])

def exact6036RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6036RawTermsValid :
    exact6036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66318⟩⟩) exact6036RawTerms (.finite 3878994884184198780231457) 6035 .exactZero (none)

def event6037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67385⟩⟩) 0 ⟨66318⟩ 6036

def event6038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67385⟩⟩) 1 ⟨67383⟩ 5820

def event6039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67385⟩⟩) (.sum [.predecessor 0 6037 .coefficient, .predecessor 1 6038 .coefficient])

def exact6040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6040RawTermsValid :
    exact6040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67385⟩⟩) exact6040RawTerms (.finite 8101376613122849735629177) 6039 .exactZero (none)

def event6041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67386⟩⟩) 0 ⟨67385⟩ 6040

def event6042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67386⟩⟩) 1 ⟨6745⟩ 5317

def event6043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67386⟩⟩) (.product (.predecessor 0 6041 .coefficient) (.predecessor 1 6042 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 5⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], []⟩, (-1)⟩)

def event6045 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 7⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩, (1)⟩)

def event6046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 8⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩, (1)⟩)

def event6047 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 9⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩)

def event6048 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 11⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩)

def event6049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 12⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩)

def event6050 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 13⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩)

def event6051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 15⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩)

def event6052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 16⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩)

def event6053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 18⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩)

def event6054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 0⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩)

def event6055 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 1⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩)

def event6056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 2⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩)

def event6057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 3⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩)

def event6058 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 4⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩)

def event6059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 6⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩)

def event6060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 10⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩)

def event6061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 14⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩)

def event6062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67386⟩⟩, .operator (⟨6040, 17⟩, ⟨5317, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩)

def exact6063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], []⟩, (1)⟩]

theorem exact6063RawTermsValid :
    exact6063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67386⟩⟩) exact6063RawTerms (.finite 40855265099867823831051625860754121276945339270267323227632632951230654388777765301824568816006679355760518649407580374777567606346378580747494525286949722600380421224393754289816792151442708922368) 6043 .exactZero (none)

def event6064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6751⟩⟩) (.authority (.factStore))

def exact6065RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩], []⟩, (1)⟩]

theorem exact6065RawTermsValid :
    exact6065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6751⟩⟩) exact6065RawTerms (.finite 913259509311826595239818521071740411827534512964575625276980100088525168305838779233307720862346840162206171719121802529293855155354423137762117884569656111648485349497) 6064 .exactZero (none)

def event6066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event6067 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event6068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 14

def event6069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 6067

def event6070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 6068 .coefficient, .predecessor 1 6069 .coefficient])

def event6071 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event6072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 6071

def event6073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 38

def event6074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 6073 .coefficient))

def event6075 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event6076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47666⟩⟩) 0 ⟨5469⟩ 6075

def event6077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47666⟩⟩) (.authority (.programFamilyFact))

def exact6078RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact6078RawTermsValid :
    exact6078RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6078 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47666⟩⟩) exact6078RawTerms (.finite 60) 6077 .exactZero (none)

def event6079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14976⟩⟩) 0 ⟨5469⟩ 6075

def event6080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14976⟩⟩) (.authority (.programFamilyFact))

def exact6081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩], []⟩, (1)⟩]

theorem exact6081RawTermsValid :
    exact6081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14976⟩⟩) exact6081RawTerms (.finite 60) 6080 .exactZero (none)

def event6082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 0 ⟨14976⟩ 6081

def event6083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 1 ⟨47666⟩ 6078

def event6084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.product (.predecessor 0 6082 .coefficient) (.predecessor 1 6083 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47667⟩⟩, .operator (⟨6081, 0⟩, ⟨6078, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩)

def exact6086RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact6086RawTermsValid :
    exact6086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6086 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47667⟩⟩) exact6086RawTerms (.finite 3600) 6084 .exactZero (none)

def event6087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47668⟩⟩) 0 ⟨47667⟩ 6086

def event6088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.identity (.predecessor 0 6087 .coefficient))

def event6089 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.finite 3600)

def event6090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48092⟩⟩) 0 ⟨47668⟩ 6089

def event6091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48092⟩⟩) (.authority (.programFamilyFact))

def exact6092RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], []⟩, (1)⟩]

theorem exact6092RawTermsValid :
    exact6092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48092⟩⟩) exact6092RawTerms (.finite 60) 6091 .exactZero (none)

def event6093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48093⟩⟩) 0 ⟨48092⟩ 6092

def event6094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.identity (.predecessor 0 6093 .coefficient))

def event6095 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.finite 60)

def event6096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48272⟩⟩) 0 ⟨48093⟩ 6095

def event6097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48272⟩⟩) (.authority (.programFamilyFact))

def exact6098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], []⟩, (1)⟩]

theorem exact6098RawTermsValid :
    exact6098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48272⟩⟩) exact6098RawTerms (.finite 63) 6097 .exactZero (none)

def event6099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44986⟩⟩) 0 ⟨5469⟩ 6075

def event6100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44986⟩⟩) (.authority (.programFamilyFact))

def exact6101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44986⟩⟩], []⟩, (1)⟩]

theorem exact6101RawTermsValid :
    exact6101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44986⟩⟩) exact6101RawTerms (.finite 58) 6100 .exactZero (none)

def event6102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14676⟩⟩) 0 ⟨5469⟩ 6075

def event6103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14676⟩⟩) (.authority (.programFamilyFact))

def exact6104RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩], []⟩, (1)⟩]

theorem exact6104RawTermsValid :
    exact6104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14676⟩⟩) exact6104RawTerms (.finite 58) 6103 .exactZero (none)

def event6105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44987⟩⟩) 0 ⟨14676⟩ 6104

def event6106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44987⟩⟩) 1 ⟨44986⟩ 6101

def event6107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44987⟩⟩) (.product (.predecessor 0 6105 .coefficient) (.predecessor 1 6106 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6108 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44987⟩⟩, .operator (⟨6104, 0⟩, ⟨6101, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], []⟩, (1)⟩)

def exact6109RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], []⟩, (1)⟩]

theorem exact6109RawTermsValid :
    exact6109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44987⟩⟩) exact6109RawTerms (.finite 3364) 6107 .exactZero (none)

def event6110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44988⟩⟩) 0 ⟨44987⟩ 6109

def event6111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44988⟩⟩) (.identity (.predecessor 0 6110 .coefficient))

def event6112 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44988⟩⟩) (.finite 3364)

def event6113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45412⟩⟩) 0 ⟨44988⟩ 6112

def event6114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45412⟩⟩) (.authority (.programFamilyFact))

def exact6115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45412⟩⟩], []⟩, (1)⟩]

theorem exact6115RawTermsValid :
    exact6115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45412⟩⟩) exact6115RawTerms (.finite 58) 6114 .exactZero (none)

def event6116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45413⟩⟩) 0 ⟨45412⟩ 6115

def event6117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45413⟩⟩) (.identity (.predecessor 0 6116 .coefficient))

def event6118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45413⟩⟩) (.finite 58)

def event6119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45592⟩⟩) 0 ⟨45413⟩ 6118

def event6120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45592⟩⟩) (.authority (.programFamilyFact))

def exact6121RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], []⟩, (1)⟩]

theorem exact6121RawTermsValid :
    exact6121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45592⟩⟩) exact6121RawTerms (.finite 63) 6120 .exactZero (none)

def event6122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 6075

def event6123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact6124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact6124RawTermsValid :
    exact6124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact6124RawTerms (.finite 52) 6123 .exactZero (none)

def event6125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 6075

def event6126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact6127RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact6127RawTermsValid :
    exact6127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6127 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact6127RawTerms (.finite 52) 6126 .exactZero (none)

def event6128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 6127

def event6129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 6124

def event6130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 6128 .coefficient) (.predecessor 1 6129 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6131 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42307⟩⟩, .operator (⟨6127, 0⟩, ⟨6124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩)

def exact6132RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact6132RawTermsValid :
    exact6132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6132 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact6132RawTerms (.finite 2704) 6130 .exactZero (none)

def event6133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 6132

def event6134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 6133 .coefficient))

def event6135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event6136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 6135

def event6137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact6138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact6138RawTermsValid :
    exact6138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact6138RawTerms (.finite 52) 6137 .exactZero (none)

def event6139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 6138

def event6140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 6139 .coefficient))

def event6141 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event6142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42908⟩⟩) 0 ⟨42733⟩ 6141

def event6143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42908⟩⟩) (.authority (.programFamilyFact))

def eventLeaf368 : Array AnnotatedEvent := #[
  { event := event5888
    frameStart := 0 },
  { event := event5889
    frameStart := 0 },
  { event := event5890
    frameStart := 0 },
  { event := event5891
    frameStart := 0 },
  { event := event5892
    frameStart := 0 },
  { event := event5893
    frameStart := 0 },
  { event := event5894
    frameStart := 0 },
  { event := event5895
    frameStart := 0 },
  { event := event5896
    frameStart := 0 },
  { event := event5897
    frameStart := 0 },
  { event := event5898
    frameStart := 0 },
  { event := event5899
    frameStart := 0 },
  { event := event5900
    frameStart := 0 },
  { event := event5901
    frameStart := 0 },
  { event := event5902
    frameStart := 0 },
  { event := event5903
    frameStart := 0 }
]

def eventLeaf369 : Array AnnotatedEvent := #[
  { event := event5904
    frameStart := 0 },
  { event := event5905
    frameStart := 0 },
  { event := event5906
    frameStart := 0 },
  { event := event5907
    frameStart := 0 },
  { event := event5908
    frameStart := 0 },
  { event := event5909
    frameStart := 0 },
  { event := event5910
    frameStart := 0 },
  { event := event5911
    frameStart := 0 },
  { event := event5912
    frameStart := 0 },
  { event := event5913
    frameStart := 0 },
  { event := event5914
    frameStart := 0 },
  { event := event5915
    frameStart := 0 },
  { event := event5916
    frameStart := 0 },
  { event := event5917
    frameStart := 0 },
  { event := event5918
    frameStart := 0 },
  { event := event5919
    frameStart := 0 }
]

def eventLeaf370 : Array AnnotatedEvent := #[
  { event := event5920
    frameStart := 0 },
  { event := event5921
    frameStart := 0 },
  { event := event5922
    frameStart := 0 },
  { event := event5923
    frameStart := 0 },
  { event := event5924
    frameStart := 0 },
  { event := event5925
    frameStart := 0 },
  { event := event5926
    frameStart := 0 },
  { event := event5927
    frameStart := 0 },
  { event := event5928
    frameStart := 0 },
  { event := event5929
    frameStart := 0 },
  { event := event5930
    frameStart := 0 },
  { event := event5931
    frameStart := 0 },
  { event := event5932
    frameStart := 0 },
  { event := event5933
    frameStart := 0 },
  { event := event5934
    frameStart := 0 },
  { event := event5935
    frameStart := 0 }
]

def eventLeaf371 : Array AnnotatedEvent := #[
  { event := event5936
    frameStart := 0 },
  { event := event5937
    frameStart := 0 },
  { event := event5938
    frameStart := 0 },
  { event := event5939
    frameStart := 0 },
  { event := event5940
    frameStart := 0 },
  { event := event5941
    frameStart := 0 },
  { event := event5942
    frameStart := 0 },
  { event := event5943
    frameStart := 0 },
  { event := event5944
    frameStart := 0 },
  { event := event5945
    frameStart := 0 },
  { event := event5946
    frameStart := 0 },
  { event := event5947
    frameStart := 0 },
  { event := event5948
    frameStart := 0 },
  { event := event5949
    frameStart := 0 },
  { event := event5950
    frameStart := 0 },
  { event := event5951
    frameStart := 0 }
]

def eventLeaf372 : Array AnnotatedEvent := #[
  { event := event5952
    frameStart := 0 },
  { event := event5953
    frameStart := 0 },
  { event := event5954
    frameStart := 0 },
  { event := event5955
    frameStart := 0 },
  { event := event5956
    frameStart := 0 },
  { event := event5957
    frameStart := 0 },
  { event := event5958
    frameStart := 0 },
  { event := event5959
    frameStart := 0 },
  { event := event5960
    frameStart := 0 },
  { event := event5961
    frameStart := 0 },
  { event := event5962
    frameStart := 0 },
  { event := event5963
    frameStart := 0 },
  { event := event5964
    frameStart := 0 },
  { event := event5965
    frameStart := 0 },
  { event := event5966
    frameStart := 0 },
  { event := event5967
    frameStart := 0 }
]

def eventLeaf373 : Array AnnotatedEvent := #[
  { event := event5968
    frameStart := 0 },
  { event := event5969
    frameStart := 0 },
  { event := event5970
    frameStart := 0 },
  { event := event5971
    frameStart := 0 },
  { event := event5972
    frameStart := 0 },
  { event := event5973
    frameStart := 0 },
  { event := event5974
    frameStart := 0 },
  { event := event5975
    frameStart := 0 },
  { event := event5976
    frameStart := 0 },
  { event := event5977
    frameStart := 0 },
  { event := event5978
    frameStart := 0 },
  { event := event5979
    frameStart := 0 },
  { event := event5980
    frameStart := 0 },
  { event := event5981
    frameStart := 0 },
  { event := event5982
    frameStart := 0 },
  { event := event5983
    frameStart := 0 }
]

def eventLeaf374 : Array AnnotatedEvent := #[
  { event := event5984
    frameStart := 0 },
  { event := event5985
    frameStart := 0 },
  { event := event5986
    frameStart := 0 },
  { event := event5987
    frameStart := 0 },
  { event := event5988
    frameStart := 0 },
  { event := event5989
    frameStart := 0 },
  { event := event5990
    frameStart := 0 },
  { event := event5991
    frameStart := 0 },
  { event := event5992
    frameStart := 0 },
  { event := event5993
    frameStart := 0 },
  { event := event5994
    frameStart := 0 },
  { event := event5995
    frameStart := 0 },
  { event := event5996
    frameStart := 0 },
  { event := event5997
    frameStart := 0 },
  { event := event5998
    frameStart := 0 },
  { event := event5999
    frameStart := 0 }
]

def eventLeaf375 : Array AnnotatedEvent := #[
  { event := event6000
    frameStart := 0 },
  { event := event6001
    frameStart := 0 },
  { event := event6002
    frameStart := 0 },
  { event := event6003
    frameStart := 0 },
  { event := event6004
    frameStart := 0 },
  { event := event6005
    frameStart := 0 },
  { event := event6006
    frameStart := 0 },
  { event := event6007
    frameStart := 0 },
  { event := event6008
    frameStart := 0 },
  { event := event6009
    frameStart := 0 },
  { event := event6010
    frameStart := 0 },
  { event := event6011
    frameStart := 0 },
  { event := event6012
    frameStart := 0 },
  { event := event6013
    frameStart := 0 },
  { event := event6014
    frameStart := 0 },
  { event := event6015
    frameStart := 0 }
]

def eventLeaf376 : Array AnnotatedEvent := #[
  { event := event6016
    frameStart := 0 },
  { event := event6017
    frameStart := 0 },
  { event := event6018
    frameStart := 0 },
  { event := event6019
    frameStart := 0 },
  { event := event6020
    frameStart := 0 },
  { event := event6021
    frameStart := 0 },
  { event := event6022
    frameStart := 0 },
  { event := event6023
    frameStart := 0 },
  { event := event6024
    frameStart := 0 },
  { event := event6025
    frameStart := 0 },
  { event := event6026
    frameStart := 0 },
  { event := event6027
    frameStart := 0 },
  { event := event6028
    frameStart := 0 },
  { event := event6029
    frameStart := 0 },
  { event := event6030
    frameStart := 0 },
  { event := event6031
    frameStart := 0 }
]

def eventLeaf377 : Array AnnotatedEvent := #[
  { event := event6032
    frameStart := 0 },
  { event := event6033
    frameStart := 0 },
  { event := event6034
    frameStart := 0 },
  { event := event6035
    frameStart := 0 },
  { event := event6036
    frameStart := 0 },
  { event := event6037
    frameStart := 0 },
  { event := event6038
    frameStart := 0 },
  { event := event6039
    frameStart := 0 },
  { event := event6040
    frameStart := 0 },
  { event := event6041
    frameStart := 0 },
  { event := event6042
    frameStart := 0 },
  { event := event6043
    frameStart := 0 },
  { event := event6044
    frameStart := 0 },
  { event := event6045
    frameStart := 0 },
  { event := event6046
    frameStart := 0 },
  { event := event6047
    frameStart := 0 }
]

def eventLeaf378 : Array AnnotatedEvent := #[
  { event := event6048
    frameStart := 0 },
  { event := event6049
    frameStart := 0 },
  { event := event6050
    frameStart := 0 },
  { event := event6051
    frameStart := 0 },
  { event := event6052
    frameStart := 0 },
  { event := event6053
    frameStart := 0 },
  { event := event6054
    frameStart := 0 },
  { event := event6055
    frameStart := 0 },
  { event := event6056
    frameStart := 0 },
  { event := event6057
    frameStart := 0 },
  { event := event6058
    frameStart := 0 },
  { event := event6059
    frameStart := 0 },
  { event := event6060
    frameStart := 0 },
  { event := event6061
    frameStart := 0 },
  { event := event6062
    frameStart := 0 },
  { event := event6063
    frameStart := 0 }
]

def eventLeaf379 : Array AnnotatedEvent := #[
  { event := event6064
    frameStart := 0 },
  { event := event6065
    frameStart := 0 },
  { event := event6066
    frameStart := 0 },
  { event := event6067
    frameStart := 0 },
  { event := event6068
    frameStart := 0 },
  { event := event6069
    frameStart := 0 },
  { event := event6070
    frameStart := 0 },
  { event := event6071
    frameStart := 0 },
  { event := event6072
    frameStart := 0 },
  { event := event6073
    frameStart := 0 },
  { event := event6074
    frameStart := 0 },
  { event := event6075
    frameStart := 0 },
  { event := event6076
    frameStart := 0 },
  { event := event6077
    frameStart := 0 },
  { event := event6078
    frameStart := 0 },
  { event := event6079
    frameStart := 0 }
]

def eventLeaf380 : Array AnnotatedEvent := #[
  { event := event6080
    frameStart := 0 },
  { event := event6081
    frameStart := 0 },
  { event := event6082
    frameStart := 0 },
  { event := event6083
    frameStart := 0 },
  { event := event6084
    frameStart := 0 },
  { event := event6085
    frameStart := 0 },
  { event := event6086
    frameStart := 0 },
  { event := event6087
    frameStart := 0 },
  { event := event6088
    frameStart := 0 },
  { event := event6089
    frameStart := 0 },
  { event := event6090
    frameStart := 0 },
  { event := event6091
    frameStart := 0 },
  { event := event6092
    frameStart := 0 },
  { event := event6093
    frameStart := 0 },
  { event := event6094
    frameStart := 0 },
  { event := event6095
    frameStart := 0 }
]

def eventLeaf381 : Array AnnotatedEvent := #[
  { event := event6096
    frameStart := 0 },
  { event := event6097
    frameStart := 0 },
  { event := event6098
    frameStart := 0 },
  { event := event6099
    frameStart := 0 },
  { event := event6100
    frameStart := 0 },
  { event := event6101
    frameStart := 0 },
  { event := event6102
    frameStart := 0 },
  { event := event6103
    frameStart := 0 },
  { event := event6104
    frameStart := 0 },
  { event := event6105
    frameStart := 0 },
  { event := event6106
    frameStart := 0 },
  { event := event6107
    frameStart := 0 },
  { event := event6108
    frameStart := 0 },
  { event := event6109
    frameStart := 0 },
  { event := event6110
    frameStart := 0 },
  { event := event6111
    frameStart := 0 }
]

def eventLeaf382 : Array AnnotatedEvent := #[
  { event := event6112
    frameStart := 0 },
  { event := event6113
    frameStart := 0 },
  { event := event6114
    frameStart := 0 },
  { event := event6115
    frameStart := 0 },
  { event := event6116
    frameStart := 0 },
  { event := event6117
    frameStart := 0 },
  { event := event6118
    frameStart := 0 },
  { event := event6119
    frameStart := 0 },
  { event := event6120
    frameStart := 0 },
  { event := event6121
    frameStart := 0 },
  { event := event6122
    frameStart := 0 },
  { event := event6123
    frameStart := 0 },
  { event := event6124
    frameStart := 0 },
  { event := event6125
    frameStart := 0 },
  { event := event6126
    frameStart := 0 },
  { event := event6127
    frameStart := 0 }
]

def eventLeaf383 : Array AnnotatedEvent := #[
  { event := event6128
    frameStart := 0 },
  { event := event6129
    frameStart := 0 },
  { event := event6130
    frameStart := 0 },
  { event := event6131
    frameStart := 0 },
  { event := event6132
    frameStart := 0 },
  { event := event6133
    frameStart := 0 },
  { event := event6134
    frameStart := 0 },
  { event := event6135
    frameStart := 0 },
  { event := event6136
    frameStart := 0 },
  { event := event6137
    frameStart := 0 },
  { event := event6138
    frameStart := 0 },
  { event := event6139
    frameStart := 0 },
  { event := event6140
    frameStart := 0 },
  { event := event6141
    frameStart := 0 },
  { event := event6142
    frameStart := 0 },
  { event := event6143
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events023
