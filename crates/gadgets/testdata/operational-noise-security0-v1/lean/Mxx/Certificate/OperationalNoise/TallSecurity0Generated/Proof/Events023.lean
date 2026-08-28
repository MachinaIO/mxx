import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events023

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event5888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7653⟩⟩) 1 ⟨7633⟩ 5827

def event5889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7653⟩⟩) (.sum [.predecessor 0 5887 .coefficient, .predecessor 1 5888 .coefficient])

def exact5890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩]

theorem exact5890RawTermsValid :
    exact5890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5890 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7653⟩⟩) exact5890RawTerms .large 5889 .exactZero (none)

def event5891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7654⟩⟩) 0 ⟨7653⟩ 5890

def event5892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7654⟩⟩) 1 ⟨7634⟩ 5807

def event5893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7654⟩⟩) (.sum [.predecessor 0 5891 .coefficient, .predecessor 1 5892 .coefficient])

def exact5894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩]

theorem exact5894RawTermsValid :
    exact5894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7654⟩⟩) exact5894RawTerms .large 5893 .exactZero (none)

def event5895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7655⟩⟩) 0 ⟨7654⟩ 5894

def event5896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7655⟩⟩) 1 ⟨7635⟩ 5787

def event5897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7655⟩⟩) (.sum [.predecessor 0 5895 .coefficient, .predecessor 1 5896 .coefficient])

def exact5898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩]

theorem exact5898RawTermsValid :
    exact5898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5898 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7655⟩⟩) exact5898RawTerms .large 5897 .exactZero (none)

def event5899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7656⟩⟩) 0 ⟨7655⟩ 5898

def event5900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7656⟩⟩) 1 ⟨7636⟩ 5767

def event5901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7656⟩⟩) (.sum [.predecessor 0 5899 .coefficient, .predecessor 1 5900 .coefficient])

def exact5902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩]

theorem exact5902RawTermsValid :
    exact5902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5902 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7656⟩⟩) exact5902RawTerms .large 5901 .exactZero (none)

def event5903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7657⟩⟩) 0 ⟨7656⟩ 5902

def event5904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7657⟩⟩) 1 ⟨7637⟩ 5747

def event5905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7657⟩⟩) (.sum [.predecessor 0 5903 .coefficient, .predecessor 1 5904 .coefficient])

def exact5906RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩]

theorem exact5906RawTermsValid :
    exact5906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5906 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7657⟩⟩) exact5906RawTerms .large 5905 .exactZero (none)

def event5907 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7658⟩⟩) 0 ⟨7657⟩ 5906

def event5908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7658⟩⟩) 1 ⟨7638⟩ 5727

def event5909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7658⟩⟩) (.sum [.predecessor 0 5907 .coefficient, .predecessor 1 5908 .coefficient])

def exact5910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩]

theorem exact5910RawTermsValid :
    exact5910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5910 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7658⟩⟩) exact5910RawTerms .large 5909 .exactZero (none)

def event5911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7659⟩⟩) 0 ⟨7658⟩ 5910

def event5912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7659⟩⟩) 1 ⟨7639⟩ 5707

def event5913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7659⟩⟩) (.sum [.predecessor 0 5911 .coefficient, .predecessor 1 5912 .coefficient])

def exact5914RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩]

theorem exact5914RawTermsValid :
    exact5914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7659⟩⟩) exact5914RawTerms .large 5913 .exactZero (none)

def event5915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7660⟩⟩) 0 ⟨7659⟩ 5914

def event5916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7660⟩⟩) 1 ⟨7640⟩ 5687

def event5917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7660⟩⟩) (.sum [.predecessor 0 5915 .coefficient, .predecessor 1 5916 .coefficient])

def exact5918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩]

theorem exact5918RawTermsValid :
    exact5918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7660⟩⟩) exact5918RawTerms .large 5917 .exactZero (none)

def event5919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7661⟩⟩) 0 ⟨7660⟩ 5918

def event5920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7661⟩⟩) 1 ⟨7641⟩ 5667

def event5921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7661⟩⟩) (.sum [.predecessor 0 5919 .coefficient, .predecessor 1 5920 .coefficient])

def exact5922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩]

theorem exact5922RawTermsValid :
    exact5922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7661⟩⟩) exact5922RawTerms .large 5921 .exactZero (none)

def event5923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7662⟩⟩) 0 ⟨7661⟩ 5922

def event5924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7662⟩⟩) 1 ⟨7642⟩ 5647

def event5925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7662⟩⟩) (.sum [.predecessor 0 5923 .coefficient, .predecessor 1 5924 .coefficient])

def exact5926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩]

theorem exact5926RawTermsValid :
    exact5926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5926 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7662⟩⟩) exact5926RawTerms .large 5925 .exactZero (none)

def event5927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7663⟩⟩) 0 ⟨7662⟩ 5926

def event5928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7663⟩⟩) 1 ⟨7643⟩ 5627

def event5929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7663⟩⟩) (.sum [.predecessor 0 5927 .coefficient, .predecessor 1 5928 .coefficient])

def exact5930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩]

theorem exact5930RawTermsValid :
    exact5930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7663⟩⟩) exact5930RawTerms .large 5929 .exactZero (none)

def event5931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7664⟩⟩) 0 ⟨7663⟩ 5930

def event5932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7664⟩⟩) 1 ⟨7644⟩ 5607

def event5933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7664⟩⟩) (.sum [.predecessor 0 5931 .coefficient, .predecessor 1 5932 .coefficient])

def exact5934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩]

theorem exact5934RawTermsValid :
    exact5934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7664⟩⟩) exact5934RawTerms .large 5933 .exactZero (none)

def event5935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7665⟩⟩) 0 ⟨7664⟩ 5934

def event5936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7665⟩⟩) 1 ⟨7645⟩ 5587

def event5937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7665⟩⟩) (.sum [.predecessor 0 5935 .coefficient, .predecessor 1 5936 .coefficient])

def exact5938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩]

theorem exact5938RawTermsValid :
    exact5938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7665⟩⟩) exact5938RawTerms .large 5937 .exactZero (none)

def event5939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7666⟩⟩) 0 ⟨7665⟩ 5938

def event5940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7666⟩⟩) 1 ⟨7646⟩ 5567

def event5941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7666⟩⟩) (.sum [.predecessor 0 5939 .coefficient, .predecessor 1 5940 .coefficient])

def exact5942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩]

theorem exact5942RawTermsValid :
    exact5942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7666⟩⟩) exact5942RawTerms .large 5941 .exactZero (none)

def event5943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7667⟩⟩) 0 ⟨7666⟩ 5942

def event5944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7667⟩⟩) 1 ⟨7647⟩ 5547

def event5945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7667⟩⟩) (.sum [.predecessor 0 5943 .coefficient, .predecessor 1 5944 .coefficient])

def exact5946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩]

theorem exact5946RawTermsValid :
    exact5946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7667⟩⟩) exact5946RawTerms .large 5945 .exactZero (none)

def event5947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7668⟩⟩) 0 ⟨7667⟩ 5946

def event5948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7668⟩⟩) 1 ⟨7648⟩ 5527

def event5949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7668⟩⟩) (.sum [.predecessor 0 5947 .coefficient, .predecessor 1 5948 .coefficient])

def exact5950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩, (1)⟩]

theorem exact5950RawTermsValid :
    exact5950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7668⟩⟩) exact5950RawTerms .large 5949 .exactZero (none)

def event5951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7795⟩⟩) 0 ⟨7668⟩ 5950

def event5952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7795⟩⟩) 1 ⟨7649⟩ 5507

def event5953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7795⟩⟩) (.sum [.predecessor 0 5951 .coefficient, .predecessor 1 5952 .coefficient])

def exact5954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩, (-1)⟩]

theorem exact5954RawTermsValid :
    exact5954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5954 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7795⟩⟩) exact5954RawTerms .large 5953 .exactZero (none)

def event5955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7885⟩⟩) 0 ⟨7795⟩ 5954

def event5956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7885⟩⟩) (.authority (.operator))

def exact5957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact5957RawTermsValid :
    exact5957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7885⟩⟩) exact5957RawTerms (.finite 8192) 5956 .exactZero (none)

def event5958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7886⟩⟩) 0 ⟨7885⟩ 5957

def event5959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7886⟩⟩) 1 ⟨2348⟩ 4

def event5960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7886⟩⟩) (.scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient)))

def exact5961RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact5961RawTermsValid :
    exact5961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7886⟩⟩) exact5961RawTerms (.finite 8192) 5960 .exactZero (none)

def event5962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6745⟩⟩) 0 ⟨6689⟩ 5477

def event5963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6745⟩⟩) (.authority (.operator))

def exact5964RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩]⟩, (1)⟩]

theorem exact5964RawTermsValid :
    exact5964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6745⟩⟩) exact5964RawTerms .large 5963 .exactZero (none)

def event5965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7887⟩⟩) 0 ⟨6745⟩ 5964

def event5966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7887⟩⟩) 1 ⟨7886⟩ 5961

def event5967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7887⟩⟩) (.product (.predecessor 0 5965 .coefficient) (.predecessor 1 5966 .coefficient) (⟨false, false, none, none, none⟩))

def event5968 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7887⟩⟩, .operator (⟨5964, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact5969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact5969RawTermsValid :
    exact5969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5969 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7887⟩⟩) exact5969RawTerms .large 5967 .exactZero (none)

def event5970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7911⟩⟩) 0 ⟨7887⟩ 5969

def event5971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7911⟩⟩) 1 ⟨7820⟩ 5487

def event5972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7911⟩⟩) (.product (.predecessor 0 5970 .coefficient) (.predecessor 1 5971 .coefficient) (⟨false, false, none, none, none⟩))

def event5973 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7911⟩⟩, .operator (⟨5969, 0⟩, ⟨5487, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩]⟩, (1)⟩)

def exact5974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩]⟩, (1)⟩]

theorem exact5974RawTermsValid :
    exact5974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7911⟩⟩) exact5974RawTerms .large 5972 .exactZero (none)

def event5975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7917⟩⟩) 0 ⟨7911⟩ 5974

def event5976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7917⟩⟩) 1 ⟨6646⟩ 5476

def event5977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7917⟩⟩) (.product (.predecessor 0 5975 .coefficient) (.predecessor 1 5976 .coefficient) (⟨false, false, none, none, none⟩))

def event5978 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7917⟩⟩, .operator (⟨5974, 0⟩, ⟨5476, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩)

def exact5979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩, (1)⟩]

theorem exact5979RawTermsValid :
    exact5979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7917⟩⟩) exact5979RawTerms .large 5977 .exactZero (none)

def event5980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6615⟩⟩) 0 ⟨6544⟩ 2

def event5981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6615⟩⟩) 1 ⟨6543⟩ 829

def event5982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6615⟩⟩) (.product (.predecessor 0 5980 .coefficient) (.predecessor 1 5981 .coefficient) (⟨false, true, none, none, some 1⟩))

def event5983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6615⟩⟩, .operator (⟨2, 0⟩, ⟨829, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6543⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact5984RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6543⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact5984RawTermsValid :
    exact5984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6615⟩⟩) exact5984RawTerms .large 5982 .exactZero (none)

def event5985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6687⟩⟩) 0 ⟨6615⟩ 5984

def event5986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6687⟩⟩) (.authority (.operator))

def exact5987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩]

theorem exact5987RawTermsValid :
    exact5987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5987 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6687⟩⟩) exact5987RawTerms (.finite 8192) 5986 .exactZero (none)

def event5988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6688⟩⟩) 0 ⟨6687⟩ 5987

def event5989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6688⟩⟩) 1 ⟨2348⟩ 4

def event5990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6688⟩⟩) (.scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient)))

def exact5991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩]

theorem exact5991RawTermsValid :
    exact5991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6688⟩⟩) exact5991RawTerms (.finite 8192) 5990 .exactZero (none)

def event5992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6748⟩⟩) 0 ⟨6689⟩ 5477

def event5993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6748⟩⟩) (.authority (.operator))

def exact5994RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩, (1)⟩]

theorem exact5994RawTermsValid :
    exact5994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5994 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6748⟩⟩) exact5994RawTerms .large 5993 .exactZero (none)

def event5995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7821⟩⟩) 0 ⟨6748⟩ 5994

def event5996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7821⟩⟩) (.authority (.operator))

def exact5997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩, (1)⟩]

theorem exact5997RawTermsValid :
    exact5997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event5997 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7821⟩⟩) exact5997RawTerms (.finite 8192) 5996 .exactZero (none)

def event5998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7822⟩⟩) 0 ⟨7821⟩ 5997

def event5999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7822⟩⟩) 1 ⟨2348⟩ 4

def event6000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7822⟩⟩) (.scale (.predecessor 0 5998 .coefficient) (.value (.predecessor 1 5999 .coefficient)))

def exact6001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩, (1)⟩]

theorem exact6001RawTermsValid :
    exact6001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7822⟩⟩) exact6001RawTerms (.finite 8192) 6000 .exactZero (none)

def event6002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6747⟩⟩) 0 ⟨6689⟩ 5477

def event6003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6747⟩⟩) (.authority (.operator))

def exact6004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩]⟩, (1)⟩]

theorem exact6004RawTermsValid :
    exact6004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6747⟩⟩) exact6004RawTerms .large 6003 .exactZero (none)

def event6005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7888⟩⟩) 0 ⟨6747⟩ 6004

def event6006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7888⟩⟩) 1 ⟨7886⟩ 5961

def event6007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7888⟩⟩) (.product (.predecessor 0 6005 .coefficient) (.predecessor 1 6006 .coefficient) (⟨false, false, none, none, none⟩))

def event6008 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7888⟩⟩, .operator (⟨6004, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact6009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact6009RawTermsValid :
    exact6009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7888⟩⟩) exact6009RawTerms .large 6007 .exactZero (none)

def event6010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7912⟩⟩) 0 ⟨7888⟩ 6009

def event6011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7912⟩⟩) 1 ⟨7822⟩ 6001

def event6012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7912⟩⟩) (.product (.predecessor 0 6010 .coefficient) (.predecessor 1 6011 .coefficient) (⟨false, false, none, none, none⟩))

def event6013 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7912⟩⟩, .operator (⟨6009, 0⟩, ⟨6001, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩, (1)⟩)

def exact6014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩, (1)⟩]

theorem exact6014RawTermsValid :
    exact6014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6014 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7912⟩⟩) exact6014RawTerms .large 6012 .exactZero (none)

def event6015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7918⟩⟩) 0 ⟨7912⟩ 6014

def event6016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7918⟩⟩) 1 ⟨6688⟩ 5991

def event6017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7918⟩⟩) (.product (.predecessor 0 6015 .coefficient) (.predecessor 1 6016 .coefficient) (⟨false, false, none, none, none⟩))

def event6018 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7918⟩⟩, .operator (⟨6014, 0⟩, ⟨5991, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩)

def exact6019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩, (1)⟩]

theorem exact6019RawTermsValid :
    exact6019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7918⟩⟩) exact6019RawTerms .large 6017 .exactZero (none)

def event6020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6598⟩⟩) 0 ⟨6544⟩ 2

def event6021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6598⟩⟩) 1 ⟨6425⟩ 1577

def event6022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6598⟩⟩) (.product (.predecessor 0 6020 .coefficient) (.predecessor 1 6021 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6598⟩⟩, .operator (⟨2, 0⟩, ⟨1577, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact6024RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6024RawTermsValid :
    exact6024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6598⟩⟩) exact6024RawTerms .large 6022 .exactZero (none)

def event6025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6653⟩⟩) 0 ⟨6598⟩ 6024

def event6026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6653⟩⟩) (.authority (.operator))

def exact6027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩]

theorem exact6027RawTermsValid :
    exact6027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6027 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6653⟩⟩) exact6027RawTerms (.finite 8192) 6026 .exactZero (none)

def event6028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6654⟩⟩) 0 ⟨6653⟩ 6027

def event6029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6654⟩⟩) 1 ⟨2348⟩ 4

def event6030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6654⟩⟩) (.scale (.predecessor 0 6028 .coefficient) (.value (.predecessor 1 6029 .coefficient)))

def exact6031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩]

theorem exact6031RawTermsValid :
    exact6031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6031 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6654⟩⟩) exact6031RawTerms (.finite 8192) 6030 .exactZero (none)

def event6032 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6750⟩⟩) 0 ⟨6689⟩ 5477

def event6033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6750⟩⟩) (.authority (.operator))

def exact6034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6750⟩⟩]⟩, (1)⟩]

theorem exact6034RawTermsValid :
    exact6034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6034 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6750⟩⟩) exact6034RawTerms .large 6033 .exactZero (none)

def event6035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7823⟩⟩) 0 ⟨6750⟩ 6034

def event6036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7823⟩⟩) (.authority (.operator))

def exact6037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7823⟩⟩]⟩, (1)⟩]

theorem exact6037RawTermsValid :
    exact6037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6037 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7823⟩⟩) exact6037RawTerms (.finite 8192) 6036 .exactZero (none)

def event6038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7824⟩⟩) 0 ⟨7823⟩ 6037

def event6039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7824⟩⟩) 1 ⟨2348⟩ 4

def event6040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7824⟩⟩) (.scale (.predecessor 0 6038 .coefficient) (.value (.predecessor 1 6039 .coefficient)))

def exact6041RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7823⟩⟩]⟩, (1)⟩]

theorem exact6041RawTermsValid :
    exact6041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6041 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7824⟩⟩) exact6041RawTerms (.finite 8192) 6040 .exactZero (none)

def event6042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6749⟩⟩) 0 ⟨6689⟩ 5477

def event6043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6749⟩⟩) (.authority (.operator))

def exact6044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩]⟩, (1)⟩]

theorem exact6044RawTermsValid :
    exact6044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6749⟩⟩) exact6044RawTerms .large 6043 .exactZero (none)

def event6045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7889⟩⟩) 0 ⟨6749⟩ 6044

def event6046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7889⟩⟩) 1 ⟨7886⟩ 5961

def event6047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7889⟩⟩) (.product (.predecessor 0 6045 .coefficient) (.predecessor 1 6046 .coefficient) (⟨false, false, none, none, none⟩))

def event6048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7889⟩⟩, .operator (⟨6044, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact6049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact6049RawTermsValid :
    exact6049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7889⟩⟩) exact6049RawTerms .large 6047 .exactZero (none)

def event6050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7913⟩⟩) 0 ⟨7889⟩ 6049

def event6051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7913⟩⟩) 1 ⟨7824⟩ 6041

def event6052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7913⟩⟩) (.product (.predecessor 0 6050 .coefficient) (.predecessor 1 6051 .coefficient) (⟨false, false, none, none, none⟩))

def event6053 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7913⟩⟩, .operator (⟨6049, 0⟩, ⟨6041, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩]⟩, (1)⟩)

def exact6054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩]⟩, (1)⟩]

theorem exact6054RawTermsValid :
    exact6054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6054 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7913⟩⟩) exact6054RawTerms .large 6052 .exactZero (none)

def event6055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7919⟩⟩) 0 ⟨7913⟩ 6054

def event6056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7919⟩⟩) 1 ⟨6654⟩ 6031

def event6057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7919⟩⟩) (.product (.predecessor 0 6055 .coefficient) (.predecessor 1 6056 .coefficient) (⟨false, false, none, none, none⟩))

def event6058 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7919⟩⟩, .operator (⟨6054, 0⟩, ⟨6031, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩)

def exact6059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩]⟩, (1)⟩]

theorem exact6059RawTermsValid :
    exact6059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7919⟩⟩) exact6059RawTerms .large 6057 .exactZero (none)

def event6060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6609⟩⟩) 0 ⟨6544⟩ 2

def event6061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6609⟩⟩) 1 ⟨6493⟩ 2325

def event6062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6609⟩⟩) (.product (.predecessor 0 6060 .coefficient) (.predecessor 1 6061 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6063 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6609⟩⟩, .operator (⟨2, 0⟩, ⟨2325, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact6064RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6493⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6064RawTermsValid :
    exact6064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6609⟩⟩) exact6064RawTerms .large 6062 .exactZero (none)

def event6065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6675⟩⟩) 0 ⟨6609⟩ 6064

def event6066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6675⟩⟩) (.authority (.operator))

def exact6067RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩]

theorem exact6067RawTermsValid :
    exact6067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6067 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6675⟩⟩) exact6067RawTerms (.finite 8192) 6066 .exactZero (none)

def event6068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6676⟩⟩) 0 ⟨6675⟩ 6067

def event6069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6676⟩⟩) 1 ⟨2348⟩ 4

def event6070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6676⟩⟩) (.scale (.predecessor 0 6068 .coefficient) (.value (.predecessor 1 6069 .coefficient)))

def exact6071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩]

theorem exact6071RawTermsValid :
    exact6071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6676⟩⟩) exact6071RawTerms (.finite 8192) 6070 .exactZero (none)

def event6072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6752⟩⟩) 0 ⟨6689⟩ 5477

def event6073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6752⟩⟩) (.authority (.operator))

def exact6074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6752⟩⟩]⟩, (1)⟩]

theorem exact6074RawTermsValid :
    exact6074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6074 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6752⟩⟩) exact6074RawTerms .large 6073 .exactZero (none)

def event6075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7825⟩⟩) 0 ⟨6752⟩ 6074

def event6076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7825⟩⟩) (.authority (.operator))

def exact6077RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩, (1)⟩]

theorem exact6077RawTermsValid :
    exact6077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6077 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7825⟩⟩) exact6077RawTerms (.finite 8192) 6076 .exactZero (none)

def event6078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7826⟩⟩) 0 ⟨7825⟩ 6077

def event6079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7826⟩⟩) 1 ⟨2348⟩ 4

def event6080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7826⟩⟩) (.scale (.predecessor 0 6078 .coefficient) (.value (.predecessor 1 6079 .coefficient)))

def exact6081RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩, (1)⟩]

theorem exact6081RawTermsValid :
    exact6081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6081 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7826⟩⟩) exact6081RawTerms (.finite 8192) 6080 .exactZero (none)

def event6082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6751⟩⟩) 0 ⟨6689⟩ 5477

def event6083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6751⟩⟩) (.authority (.operator))

def exact6084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩]⟩, (1)⟩]

theorem exact6084RawTermsValid :
    exact6084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6084 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6751⟩⟩) exact6084RawTerms .large 6083 .exactZero (none)

def event6085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7890⟩⟩) 0 ⟨6751⟩ 6084

def event6086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7890⟩⟩) 1 ⟨7886⟩ 5961

def event6087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7890⟩⟩) (.product (.predecessor 0 6085 .coefficient) (.predecessor 1 6086 .coefficient) (⟨false, false, none, none, none⟩))

def event6088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7890⟩⟩, .operator (⟨6084, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact6089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact6089RawTermsValid :
    exact6089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7890⟩⟩) exact6089RawTerms .large 6087 .exactZero (none)

def event6090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7914⟩⟩) 0 ⟨7890⟩ 6089

def event6091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7914⟩⟩) 1 ⟨7826⟩ 6081

def event6092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7914⟩⟩) (.product (.predecessor 0 6090 .coefficient) (.predecessor 1 6091 .coefficient) (⟨false, false, none, none, none⟩))

def event6093 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7914⟩⟩, .operator (⟨6089, 0⟩, ⟨6081, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩, (1)⟩)

def exact6094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩, (1)⟩]

theorem exact6094RawTermsValid :
    exact6094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6094 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7914⟩⟩) exact6094RawTerms .large 6092 .exactZero (none)

def event6095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7920⟩⟩) 0 ⟨7914⟩ 6094

def event6096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7920⟩⟩) 1 ⟨6676⟩ 6071

def event6097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7920⟩⟩) (.product (.predecessor 0 6095 .coefficient) (.predecessor 1 6096 .coefficient) (⟨false, false, none, none, none⟩))

def event6098 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7920⟩⟩, .operator (⟨6094, 0⟩, ⟨6071, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩)

def exact6099RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩]⟩, (1)⟩]

theorem exact6099RawTermsValid :
    exact6099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6099 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7920⟩⟩) exact6099RawTerms .large 6097 .exactZero (none)

def event6100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6614⟩⟩) 0 ⟨6544⟩ 2

def event6101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6614⟩⟩) 1 ⟨6542⟩ 3073

def event6102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6614⟩⟩) (.product (.predecessor 0 6100 .coefficient) (.predecessor 1 6101 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6103 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6614⟩⟩, .operator (⟨2, 0⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact6104RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨6542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact6104RawTermsValid :
    exact6104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6614⟩⟩) exact6104RawTerms .large 6102 .exactZero (none)

def event6105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6685⟩⟩) 0 ⟨6614⟩ 6104

def event6106 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6685⟩⟩) (.authority (.operator))

def exact6107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩]

theorem exact6107RawTermsValid :
    exact6107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6107 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6685⟩⟩) exact6107RawTerms (.finite 8192) 6106 .exactZero (none)

def event6108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6686⟩⟩) 0 ⟨6685⟩ 6107

def event6109 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6686⟩⟩) 1 ⟨2348⟩ 4

def event6110 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6686⟩⟩) (.scale (.predecessor 0 6108 .coefficient) (.value (.predecessor 1 6109 .coefficient)))

def exact6111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩]

theorem exact6111RawTermsValid :
    exact6111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6686⟩⟩) exact6111RawTerms (.finite 8192) 6110 .exactZero (none)

def event6112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6754⟩⟩) 0 ⟨6689⟩ 5477

def event6113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6754⟩⟩) (.authority (.operator))

def exact6114RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6754⟩⟩]⟩, (1)⟩]

theorem exact6114RawTermsValid :
    exact6114RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6114 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6754⟩⟩) exact6114RawTerms .large 6113 .exactZero (none)

def event6115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7827⟩⟩) 0 ⟨6754⟩ 6114

def event6116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7827⟩⟩) (.authority (.operator))

def exact6117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩, (1)⟩]

theorem exact6117RawTermsValid :
    exact6117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6117 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7827⟩⟩) exact6117RawTerms (.finite 8192) 6116 .exactZero (none)

def event6118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7828⟩⟩) 0 ⟨7827⟩ 6117

def event6119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7828⟩⟩) 1 ⟨2348⟩ 4

def event6120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7828⟩⟩) (.scale (.predecessor 0 6118 .coefficient) (.value (.predecessor 1 6119 .coefficient)))

def exact6121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩, (1)⟩]

theorem exact6121RawTermsValid :
    exact6121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6121 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7828⟩⟩) exact6121RawTerms (.finite 8192) 6120 .exactZero (none)

def event6122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6753⟩⟩) 0 ⟨6689⟩ 5477

def event6123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6753⟩⟩) (.authority (.operator))

def exact6124RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩]⟩, (1)⟩]

theorem exact6124RawTermsValid :
    exact6124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6753⟩⟩) exact6124RawTerms .large 6123 .exactZero (none)

def event6125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7891⟩⟩) 0 ⟨6753⟩ 6124

def event6126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7891⟩⟩) 1 ⟨7886⟩ 5961

def event6127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7891⟩⟩) (.product (.predecessor 0 6125 .coefficient) (.predecessor 1 6126 .coefficient) (⟨false, false, none, none, none⟩))

def event6128 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7891⟩⟩, .operator (⟨6124, 0⟩, ⟨5961, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩)

def exact6129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩, (1)⟩]

theorem exact6129RawTermsValid :
    exact6129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7891⟩⟩) exact6129RawTerms .large 6127 .exactZero (none)

def event6130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7915⟩⟩) 0 ⟨7891⟩ 6129

def event6131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7915⟩⟩) 1 ⟨7828⟩ 6121

def event6132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7915⟩⟩) (.product (.predecessor 0 6130 .coefficient) (.predecessor 1 6131 .coefficient) (⟨false, false, none, none, none⟩))

def event6133 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7915⟩⟩, .operator (⟨6129, 0⟩, ⟨6121, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩, (1)⟩)

def exact6134RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩]⟩, (1)⟩]

theorem exact6134RawTermsValid :
    exact6134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6134 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7915⟩⟩) exact6134RawTerms .large 6132 .exactZero (none)

def event6135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7921⟩⟩) 0 ⟨7915⟩ 6134

def event6136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7921⟩⟩) 1 ⟨6686⟩ 6111

def event6137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7921⟩⟩) (.product (.predecessor 0 6135 .coefficient) (.predecessor 1 6136 .coefficient) (⟨false, false, none, none, none⟩))

def event6138 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7921⟩⟩, .operator (⟨6134, 0⟩, ⟨6111, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩)

def exact6139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩]⟩, (1)⟩]

theorem exact6139RawTermsValid :
    exact6139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7921⟩⟩) exact6139RawTerms .large 6137 .exactZero (none)

def event6140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6613⟩⟩) 0 ⟨6544⟩ 2

def event6141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6613⟩⟩) 1 ⟨6503⟩ 3821

def event6142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6613⟩⟩) (.product (.predecessor 0 6140 .coefficient) (.predecessor 1 6141 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6613⟩⟩, .operator (⟨2, 0⟩, ⟨3821, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨6503⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

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

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events023
