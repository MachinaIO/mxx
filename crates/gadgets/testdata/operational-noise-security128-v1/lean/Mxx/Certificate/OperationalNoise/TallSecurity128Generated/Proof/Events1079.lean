import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1079

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event276224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67306⟩⟩) 0 ⟨7233⟩ 276223

def event276225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67306⟩⟩) 1 ⟨67302⟩ 276220

def event276226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67306⟩⟩) (.sum [.predecessor 0 276224 .coefficient, .predecessor 1 276225 .coefficient])

def exact276227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276227RawTermsValid :
    exact276227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67306⟩⟩) exact276227RawTerms .large 276226 .exactZero (none)

def event276228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70984⟩⟩) 0 ⟨67306⟩ 276227

def event276229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70984⟩⟩) 1 ⟨70980⟩ 276212

def event276230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70984⟩⟩) (.sum [.predecessor 0 276228 .coefficient, .predecessor 1 276229 .coefficient])

def exact276231RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276231RawTermsValid :
    exact276231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70984⟩⟩) exact276231RawTerms .large 276230 .exactZero (none)

def event276232 : Event := .preFoldPolynomial 276231 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact276233RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event276233 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨70984⟩⟩) 276232 exact276233RawTerms .large 276230 .exactZero (none)

def event276234 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨66029⟩⟩) ⟨⟨1⟩, ⟨95⟩, ⟨135⟩⟩ ⟨274872, 276234⟩

def event276235 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68290⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (1) 0 2 (.universal 276234 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (none) 276233)

def event276236 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 18, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩)

def event276237 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 17, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276238 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 16, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276239 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 15, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276240 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 14, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276241 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 13, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 12, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 11, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276244 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 10, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276245 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 9, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276246 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 8, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276247 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 7, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276248 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 6, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276249 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 5, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276250 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 4, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276251 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276252 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276254 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 30, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 29, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 28, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 27, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 26, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 25, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276261 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 23, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276262 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 22, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276263 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 36, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276264 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 35, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 34, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276266 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 33, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276267 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 32, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276268 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 31, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276269 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 24, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276270 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 21, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276271 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 20, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276272 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 19, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩)

def event276273 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .relation 276235 37, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact276274RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276274RawTermsValid :
    exact276274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68290⟩⟩) exact276274RawTerms .large 274868 (.finite 202072841853861888) (some (274870))

def event276275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70982⟩⟩) 0 ⟨68290⟩ 276274

def event276276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70982⟩⟩) 1 ⟨70981⟩ 274858

def event276277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70982⟩⟩) (.sum [.predecessor 0 276275 .coefficient, .predecessor 1 276276 .coefficient])

def event276278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 17⟩, ⟨274858, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276279 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 30⟩, ⟨274858, 29⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276280 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 16⟩, ⟨274858, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 29⟩, ⟨274858, 28⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276282 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 15⟩, ⟨274858, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276283 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 28⟩, ⟨274858, 27⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276284 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 14⟩, ⟨274858, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276285 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 27⟩, ⟨274858, 26⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 13⟩, ⟨274858, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276287 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 26⟩, ⟨274858, 25⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 12⟩, ⟨274858, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 25⟩, ⟨274858, 24⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 11⟩, ⟨274858, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 23⟩, ⟨274858, 22⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 10⟩, ⟨274858, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 22⟩, ⟨274858, 21⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 9⟩, ⟨274858, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 36⟩, ⟨274858, 35⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 8⟩, ⟨274858, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 35⟩, ⟨274858, 34⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 7⟩, ⟨274858, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276299 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 34⟩, ⟨274858, 33⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276300 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 6⟩, ⟨274858, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 33⟩, ⟨274858, 32⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 5⟩, ⟨274858, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 32⟩, ⟨274858, 31⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 4⟩, ⟨274858, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 31⟩, ⟨274858, 30⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 3⟩, ⟨274858, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 24⟩, ⟨274858, 23⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276308 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 2⟩, ⟨274858, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 21⟩, ⟨274858, 20⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 1⟩, ⟨274858, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 20⟩, ⟨274858, 19⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 0⟩, ⟨274858, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70982⟩⟩, .operator (⟨276274, 19⟩, ⟨274858, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70982⟩⟩) (.sum [.result 276274 .summary, .result 274858 .summary])

def exact276315RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276315RawTermsValid :
    exact276315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70982⟩⟩) exact276315RawTerms .large 276277 (.finite 6221717896068416040249469506489977540968448) (some (276314))

def event276316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70983⟩⟩) 0 ⟨70982⟩ 276315

def event276317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70983⟩⟩) 1 ⟨7140⟩ 15522

def event276318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70983⟩⟩) (.product (.predecessor 0 276316 .coefficient) (.predecessor 1 276317 .coefficient) (⟨false, false, none, none, none⟩))

def event276319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70983⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) [⟨.result 15518 .coefficient, false, none⟩])

def event276320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70983⟩⟩) (.product (.result 276315 .summary) (.transfer 276319) (⟨false, false, none, none, none⟩))

def event276321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70983⟩⟩, .operator (⟨276315, 0⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event276322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70983⟩⟩, .operator (⟨276315, 1⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event276323 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70983⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515)

def event276324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70983⟩⟩, .relation 276323 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact276325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276325RawTermsValid :
    exact276325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70983⟩⟩) exact276325RawTerms .large 276318 (.finite 66805187221379434678483228029309283225584960819691520) (some (276320))

def event276326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49225⟩⟩) 0 ⟨7177⟩ 15500

def event276327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49225⟩⟩) 1 ⟨49224⟩ 266006

def event276328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49225⟩⟩) (.authority (.operator))

def exact276329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩]

theorem exact276329RawTermsValid :
    exact276329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49225⟩⟩) exact276329RawTerms .large 276328 .exactZero (none)

def event276330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49816⟩⟩) 0 ⟨49225⟩ 276329

def event276331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49816⟩⟩) (.authority (.operator))

def exact276332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩]

theorem exact276332RawTermsValid :
    exact276332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49816⟩⟩) exact276332RawTerms (.finite 8192) 276331 .exactZero (none)

def event276333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49818⟩⟩) 0 ⟨49570⟩ 266306

def event276334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49818⟩⟩) 1 ⟨49816⟩ 276332

def event276335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49818⟩⟩) (.product (.predecessor 0 276333 .coefficient) (.predecessor 1 276334 .coefficient) (⟨false, false, none, none, none⟩))

def event276336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49818⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩) [⟨.result 276332 .coefficient, false, none⟩])

def event276337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49818⟩⟩) (.product (.result 266306 .summary) (.transfer 276336) (⟨false, false, none, none, none⟩))

def event276338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49818⟩⟩, .operator (⟨266306, 0⟩, ⟨276332, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩)

def event276339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49818⟩⟩, .operator (⟨266306, 1⟩, ⟨276332, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩)

def event276340 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49818⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49816⟩⟩) ⟨49225⟩ 276329)

def event276341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49818⟩⟩, .relation 276340 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (-1)⟩)

def exact276342RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (-1)⟩]

theorem exact276342RawTermsValid :
    exact276342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49818⟩⟩) exact276342RawTerms .large 276335 (.finite 32194504275408438756654574469120) (some (276337))

def event276343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48726⟩⟩) 0 ⟨48083⟩ 12827

def event276344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48726⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact276345RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩]

theorem exact276345RawTermsValid :
    exact276345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48726⟩⟩) exact276345RawTerms (.finite 5647228698) 276344 .exactZero (none)

def event276346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48728⟩⟩) 0 ⟨48726⟩ 276345

def event276347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48728⟩⟩) 1 ⟨2370⟩ 4

def event276348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48728⟩⟩) (.scale (.predecessor 0 276346 .coefficient) (.value (.predecessor 1 276347 .coefficient)))

def exact276349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩]

theorem exact276349RawTermsValid :
    exact276349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48728⟩⟩) exact276349RawTerms (.finite 5647228698) 276348 .exactZero (none)

def event276350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48729⟩⟩) 0 ⟨5449⟩ 266120

def event276351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48729⟩⟩) 1 ⟨48728⟩ 276349

def event276352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48729⟩⟩) (.product (.predecessor 0 276350 .coefficient) (.predecessor 1 276351 .coefficient) (⟨false, false, none, none, none⟩))

def event276353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48729⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩) [⟨.result 276345 .coefficient, false, none⟩])

def event276354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48729⟩⟩) (.product (.result 266120 .summary) (.transfer 276353) (⟨false, false, none, none, none⟩))

def event276355 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48729⟩⟩, .operator (⟨266120, 0⟩, ⟨276349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩)

def event276356 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48727⟩⟩)

def event276357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event276358 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event276359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event276360 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event276361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event276362 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event276363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event276364 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event276365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 276364

def event276366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 276362

def event276367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 276365 .coefficient) (.value (.predecessor 1 276366 .coefficient)))

def event276368 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event276369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 276368

def event276370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 276360

def event276371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 276369 .coefficient, .predecessor 1 276370 .coefficient])

def event276372 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event276373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 276372

def event276374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 276358

def event276375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 276374 .coefficient))

def event276376 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event276377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47634⟩⟩) 0 ⟨5445⟩ 276376

def event276378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47634⟩⟩) (.authority (.programFamilyFact))

def exact276379RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩, (1)⟩]

theorem exact276379RawTermsValid :
    exact276379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47634⟩⟩) exact276379RawTerms (.finite 60) 276378 .exactZero (none)

def event276380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14956⟩⟩) 0 ⟨5445⟩ 276376

def event276381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14956⟩⟩) (.authority (.programFamilyFact))

def exact276382RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩], []⟩, (1)⟩]

theorem exact276382RawTermsValid :
    exact276382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14956⟩⟩) exact276382RawTerms (.finite 60) 276381 .exactZero (none)

def event276383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 0 ⟨14956⟩ 276382

def event276384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 1 ⟨47634⟩ 276379

def event276385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47635⟩⟩) (.product (.predecessor 0 276383 .coefficient) (.predecessor 1 276384 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event276386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47635⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩) [⟨.result 276382 .coefficient, true, some 1⟩, ⟨.result 276379 .coefficient, true, some 1⟩])

def event276387 : Event := .survivorFold (1) 276386

def exact276388RawTerms : List Term := []

theorem exact276388RawTermsValid :
    exact276388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47635⟩⟩) exact276388RawTerms (.finite 3600) 276385 (.finite 3600) (some (276386))

def event276389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47636⟩⟩) 0 ⟨47635⟩ 276388

def event276390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.identity (.predecessor 0 276389 .coefficient))

def event276391 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.finite 3600)

def event276392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48082⟩⟩) 0 ⟨47636⟩ 276391

def event276393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48082⟩⟩) (.authority (.programFamilyFact))

def exact276394RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], []⟩, (1)⟩]

theorem exact276394RawTermsValid :
    exact276394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48082⟩⟩) exact276394RawTerms (.finite 60) 276393 .exactZero (none)

def event276395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48083⟩⟩) 0 ⟨48082⟩ 276394

def event276396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.identity (.predecessor 0 276395 .coefficient))

def event276397 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.finite 60)

def event276398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48726⟩⟩) 0 ⟨48083⟩ 276397

def event276399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48726⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact276400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩]

theorem exact276400RawTermsValid :
    exact276400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48726⟩⟩) exact276400RawTerms (.finite 5647228698) 276399 .exactZero (none)

def event276401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact276402RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact276402RawTermsValid :
    exact276402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact276402RawTerms .large 276401 .exactZero (none)

def event276403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48727⟩⟩) 0 ⟨35⟩ 276402

def event276404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48727⟩⟩) 1 ⟨48726⟩ 276400

def event276405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48727⟩⟩) (.product (.predecessor 0 276403 .coefficient) (.predecessor 1 276404 .coefficient) (⟨false, false, none, none, none⟩))

def event276406 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48727⟩⟩, .operator (⟨276402, 0⟩, ⟨276400, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩)

def exact276407RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩]

theorem exact276407RawTermsValid :
    exact276407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276407 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48727⟩⟩) exact276407RawTerms .large 276405 .exactZero (none)

def event276408 : Event := .preFoldPolynomial 276407 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩] .exactZero none

def exact276409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩, (1)⟩]

def event276409 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48727⟩⟩) 276408 exact276409RawTerms .large 276405 .exactZero (none)

def event276410 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49821⟩⟩)

def event276411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event276412 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event276413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event276414 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event276415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event276416 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event276417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event276418 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event276419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 276418

def event276420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 276416

def event276421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 276419 .coefficient) (.value (.predecessor 1 276420 .coefficient)))

def event276422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event276423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 276422

def event276424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 276414

def event276425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 276423 .coefficient, .predecessor 1 276424 .coefficient])

def event276426 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event276427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 276426

def event276428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 276412

def event276429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 276428 .coefficient))

def event276430 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event276431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47634⟩⟩) 0 ⟨5445⟩ 276430

def event276432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47634⟩⟩) (.authority (.programFamilyFact))

def exact276433RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩, (1)⟩]

theorem exact276433RawTermsValid :
    exact276433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47634⟩⟩) exact276433RawTerms (.finite 60) 276432 .exactZero (none)

def event276434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14956⟩⟩) 0 ⟨5445⟩ 276430

def event276435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14956⟩⟩) (.authority (.programFamilyFact))

def exact276436RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩], []⟩, (1)⟩]

theorem exact276436RawTermsValid :
    exact276436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276436 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14956⟩⟩) exact276436RawTerms (.finite 60) 276435 .exactZero (none)

def event276437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 0 ⟨14956⟩ 276436

def event276438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 1 ⟨47634⟩ 276433

def event276439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47635⟩⟩) (.product (.predecessor 0 276437 .coefficient) (.predecessor 1 276438 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event276440 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47635⟩⟩, .operator (⟨276436, 0⟩, ⟨276433, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩, (1)⟩)

def exact276441RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩, (1)⟩]

theorem exact276441RawTermsValid :
    exact276441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276441 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47635⟩⟩) exact276441RawTerms (.finite 3600) 276439 .exactZero (none)

def event276442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47636⟩⟩) 0 ⟨47635⟩ 276441

def event276443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.identity (.predecessor 0 276442 .coefficient))

def event276444 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.finite 3600)

def event276445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48082⟩⟩) 0 ⟨47636⟩ 276444

def event276446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48082⟩⟩) (.authority (.programFamilyFact))

def exact276447RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], []⟩, (1)⟩]

theorem exact276447RawTermsValid :
    exact276447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48082⟩⟩) exact276447RawTerms (.finite 60) 276446 .exactZero (none)

def event276448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48083⟩⟩) 0 ⟨48082⟩ 276447

def event276449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.identity (.predecessor 0 276448 .coefficient))

def event276450 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.finite 60)

def event276451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49224⟩⟩) 0 ⟨48083⟩ 276450

def event276452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49224⟩⟩) (.authority (.programFamilyFact))

def event276453 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49224⟩⟩) (.finite 3720)

def event276454 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event276455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49225⟩⟩) 0 ⟨7177⟩ 276454

def event276456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49225⟩⟩) 1 ⟨49224⟩ 276453

def event276457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49225⟩⟩) (.authority (.operator))

def exact276458RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩]

theorem exact276458RawTermsValid :
    exact276458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49225⟩⟩) exact276458RawTerms .large 276457 .exactZero (none)

def event276459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49816⟩⟩) 0 ⟨49225⟩ 276458

def event276460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49816⟩⟩) (.authority (.operator))

def exact276461RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩]

theorem exact276461RawTermsValid :
    exact276461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276461 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49816⟩⟩) exact276461RawTerms (.finite 8192) 276460 .exactZero (none)

def event276462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event276463 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event276464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49474⟩⟩) 0 ⟨48083⟩ 276450

def event276465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49474⟩⟩) 1 ⟨136⟩ 276463

def event276466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49474⟩⟩) (.sum [.predecessor 0 276464 .coefficient, .predecessor 1 276465 .coefficient])

def event276467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49474⟩⟩) (.finite 60)

def event276468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49475⟩⟩) 0 ⟨49474⟩ 276467

def event276469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49475⟩⟩) (.identity (.predecessor 0 276468 .coefficient))

def exact276470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], []⟩, (1)⟩]

theorem exact276470RawTermsValid :
    exact276470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49475⟩⟩) exact276470RawTerms (.finite 60) 276469 .exactZero (none)

def event276471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact276472RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276472RawTermsValid :
    exact276472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact276472RawTerms .large 276471 .exactZero (none)

def event276473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49476⟩⟩) 0 ⟨6908⟩ 276472

def event276474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49476⟩⟩) 1 ⟨49475⟩ 276470

def event276475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49476⟩⟩) (.product (.predecessor 0 276473 .coefficient) (.predecessor 1 276474 .coefficient) (⟨false, false, none, none, none⟩))

def event276476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49476⟩⟩, .operator (⟨276472, 0⟩, ⟨276470, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276477RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276477RawTermsValid :
    exact276477RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276477 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49476⟩⟩) exact276477RawTerms .large 276475 .exactZero (none)

def event276478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 276454

def event276479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def eventLeaf17264 : Array AnnotatedEvent := #[
  { event := event276224
    frameStart := 275461 },
  { event := event276225
    frameStart := 275461 },
  { event := event276226
    frameStart := 275461 },
  { event := event276227
    frameStart := 275461 },
  { event := event276228
    frameStart := 275461 },
  { event := event276229
    frameStart := 275461 },
  { event := event276230
    frameStart := 275461 },
  { event := event276231
    frameStart := 275461 },
  { event := event276232
    frameStart := 275461 },
  { event := event276233
    frameStart := 275461 },
  { event := event276234
    frameStart := 0 },
  { event := event276235
    frameStart := 0 },
  { event := event276236
    frameStart := 0 },
  { event := event276237
    frameStart := 0 },
  { event := event276238
    frameStart := 0 },
  { event := event276239
    frameStart := 0 }
]

def eventLeaf17265 : Array AnnotatedEvent := #[
  { event := event276240
    frameStart := 0 },
  { event := event276241
    frameStart := 0 },
  { event := event276242
    frameStart := 0 },
  { event := event276243
    frameStart := 0 },
  { event := event276244
    frameStart := 0 },
  { event := event276245
    frameStart := 0 },
  { event := event276246
    frameStart := 0 },
  { event := event276247
    frameStart := 0 },
  { event := event276248
    frameStart := 0 },
  { event := event276249
    frameStart := 0 },
  { event := event276250
    frameStart := 0 },
  { event := event276251
    frameStart := 0 },
  { event := event276252
    frameStart := 0 },
  { event := event276253
    frameStart := 0 },
  { event := event276254
    frameStart := 0 },
  { event := event276255
    frameStart := 0 }
]

def eventLeaf17266 : Array AnnotatedEvent := #[
  { event := event276256
    frameStart := 0 },
  { event := event276257
    frameStart := 0 },
  { event := event276258
    frameStart := 0 },
  { event := event276259
    frameStart := 0 },
  { event := event276260
    frameStart := 0 },
  { event := event276261
    frameStart := 0 },
  { event := event276262
    frameStart := 0 },
  { event := event276263
    frameStart := 0 },
  { event := event276264
    frameStart := 0 },
  { event := event276265
    frameStart := 0 },
  { event := event276266
    frameStart := 0 },
  { event := event276267
    frameStart := 0 },
  { event := event276268
    frameStart := 0 },
  { event := event276269
    frameStart := 0 },
  { event := event276270
    frameStart := 0 },
  { event := event276271
    frameStart := 0 }
]

def eventLeaf17267 : Array AnnotatedEvent := #[
  { event := event276272
    frameStart := 0 },
  { event := event276273
    frameStart := 0 },
  { event := event276274
    frameStart := 0 },
  { event := event276275
    frameStart := 0 },
  { event := event276276
    frameStart := 0 },
  { event := event276277
    frameStart := 0 },
  { event := event276278
    frameStart := 0 },
  { event := event276279
    frameStart := 0 },
  { event := event276280
    frameStart := 0 },
  { event := event276281
    frameStart := 0 },
  { event := event276282
    frameStart := 0 },
  { event := event276283
    frameStart := 0 },
  { event := event276284
    frameStart := 0 },
  { event := event276285
    frameStart := 0 },
  { event := event276286
    frameStart := 0 },
  { event := event276287
    frameStart := 0 }
]

def eventLeaf17268 : Array AnnotatedEvent := #[
  { event := event276288
    frameStart := 0 },
  { event := event276289
    frameStart := 0 },
  { event := event276290
    frameStart := 0 },
  { event := event276291
    frameStart := 0 },
  { event := event276292
    frameStart := 0 },
  { event := event276293
    frameStart := 0 },
  { event := event276294
    frameStart := 0 },
  { event := event276295
    frameStart := 0 },
  { event := event276296
    frameStart := 0 },
  { event := event276297
    frameStart := 0 },
  { event := event276298
    frameStart := 0 },
  { event := event276299
    frameStart := 0 },
  { event := event276300
    frameStart := 0 },
  { event := event276301
    frameStart := 0 },
  { event := event276302
    frameStart := 0 },
  { event := event276303
    frameStart := 0 }
]

def eventLeaf17269 : Array AnnotatedEvent := #[
  { event := event276304
    frameStart := 0 },
  { event := event276305
    frameStart := 0 },
  { event := event276306
    frameStart := 0 },
  { event := event276307
    frameStart := 0 },
  { event := event276308
    frameStart := 0 },
  { event := event276309
    frameStart := 0 },
  { event := event276310
    frameStart := 0 },
  { event := event276311
    frameStart := 0 },
  { event := event276312
    frameStart := 0 },
  { event := event276313
    frameStart := 0 },
  { event := event276314
    frameStart := 0 },
  { event := event276315
    frameStart := 0 },
  { event := event276316
    frameStart := 0 },
  { event := event276317
    frameStart := 0 },
  { event := event276318
    frameStart := 0 },
  { event := event276319
    frameStart := 0 }
]

def eventLeaf17270 : Array AnnotatedEvent := #[
  { event := event276320
    frameStart := 0 },
  { event := event276321
    frameStart := 0 },
  { event := event276322
    frameStart := 0 },
  { event := event276323
    frameStart := 0 },
  { event := event276324
    frameStart := 0 },
  { event := event276325
    frameStart := 0 },
  { event := event276326
    frameStart := 0 },
  { event := event276327
    frameStart := 0 },
  { event := event276328
    frameStart := 0 },
  { event := event276329
    frameStart := 0 },
  { event := event276330
    frameStart := 0 },
  { event := event276331
    frameStart := 0 },
  { event := event276332
    frameStart := 0 },
  { event := event276333
    frameStart := 0 },
  { event := event276334
    frameStart := 0 },
  { event := event276335
    frameStart := 0 }
]

def eventLeaf17271 : Array AnnotatedEvent := #[
  { event := event276336
    frameStart := 0 },
  { event := event276337
    frameStart := 0 },
  { event := event276338
    frameStart := 0 },
  { event := event276339
    frameStart := 0 },
  { event := event276340
    frameStart := 0 },
  { event := event276341
    frameStart := 0 },
  { event := event276342
    frameStart := 0 },
  { event := event276343
    frameStart := 0 },
  { event := event276344
    frameStart := 0 },
  { event := event276345
    frameStart := 0 },
  { event := event276346
    frameStart := 0 },
  { event := event276347
    frameStart := 0 },
  { event := event276348
    frameStart := 0 },
  { event := event276349
    frameStart := 0 },
  { event := event276350
    frameStart := 0 },
  { event := event276351
    frameStart := 0 }
]

def eventLeaf17272 : Array AnnotatedEvent := #[
  { event := event276352
    frameStart := 0 },
  { event := event276353
    frameStart := 0 },
  { event := event276354
    frameStart := 0 },
  { event := event276355
    frameStart := 0 },
  { event := event276356
    frameStart := 276356 },
  { event := event276357
    frameStart := 276356 },
  { event := event276358
    frameStart := 276356 },
  { event := event276359
    frameStart := 276356 },
  { event := event276360
    frameStart := 276356 },
  { event := event276361
    frameStart := 276356 },
  { event := event276362
    frameStart := 276356 },
  { event := event276363
    frameStart := 276356 },
  { event := event276364
    frameStart := 276356 },
  { event := event276365
    frameStart := 276356 },
  { event := event276366
    frameStart := 276356 },
  { event := event276367
    frameStart := 276356 }
]

def eventLeaf17273 : Array AnnotatedEvent := #[
  { event := event276368
    frameStart := 276356 },
  { event := event276369
    frameStart := 276356 },
  { event := event276370
    frameStart := 276356 },
  { event := event276371
    frameStart := 276356 },
  { event := event276372
    frameStart := 276356 },
  { event := event276373
    frameStart := 276356 },
  { event := event276374
    frameStart := 276356 },
  { event := event276375
    frameStart := 276356 },
  { event := event276376
    frameStart := 276356 },
  { event := event276377
    frameStart := 276356 },
  { event := event276378
    frameStart := 276356 },
  { event := event276379
    frameStart := 276356 },
  { event := event276380
    frameStart := 276356 },
  { event := event276381
    frameStart := 276356 },
  { event := event276382
    frameStart := 276356 },
  { event := event276383
    frameStart := 276356 }
]

def eventLeaf17274 : Array AnnotatedEvent := #[
  { event := event276384
    frameStart := 276356 },
  { event := event276385
    frameStart := 276356 },
  { event := event276386
    frameStart := 276356 },
  { event := event276387
    frameStart := 276356 },
  { event := event276388
    frameStart := 276356 },
  { event := event276389
    frameStart := 276356 },
  { event := event276390
    frameStart := 276356 },
  { event := event276391
    frameStart := 276356 },
  { event := event276392
    frameStart := 276356 },
  { event := event276393
    frameStart := 276356 },
  { event := event276394
    frameStart := 276356 },
  { event := event276395
    frameStart := 276356 },
  { event := event276396
    frameStart := 276356 },
  { event := event276397
    frameStart := 276356 },
  { event := event276398
    frameStart := 276356 },
  { event := event276399
    frameStart := 276356 }
]

def eventLeaf17275 : Array AnnotatedEvent := #[
  { event := event276400
    frameStart := 276356 },
  { event := event276401
    frameStart := 276356 },
  { event := event276402
    frameStart := 276356 },
  { event := event276403
    frameStart := 276356 },
  { event := event276404
    frameStart := 276356 },
  { event := event276405
    frameStart := 276356 },
  { event := event276406
    frameStart := 276356 },
  { event := event276407
    frameStart := 276356 },
  { event := event276408
    frameStart := 276356 },
  { event := event276409
    frameStart := 276356 },
  { event := event276410
    frameStart := 276410 },
  { event := event276411
    frameStart := 276410 },
  { event := event276412
    frameStart := 276410 },
  { event := event276413
    frameStart := 276410 },
  { event := event276414
    frameStart := 276410 },
  { event := event276415
    frameStart := 276410 }
]

def eventLeaf17276 : Array AnnotatedEvent := #[
  { event := event276416
    frameStart := 276410 },
  { event := event276417
    frameStart := 276410 },
  { event := event276418
    frameStart := 276410 },
  { event := event276419
    frameStart := 276410 },
  { event := event276420
    frameStart := 276410 },
  { event := event276421
    frameStart := 276410 },
  { event := event276422
    frameStart := 276410 },
  { event := event276423
    frameStart := 276410 },
  { event := event276424
    frameStart := 276410 },
  { event := event276425
    frameStart := 276410 },
  { event := event276426
    frameStart := 276410 },
  { event := event276427
    frameStart := 276410 },
  { event := event276428
    frameStart := 276410 },
  { event := event276429
    frameStart := 276410 },
  { event := event276430
    frameStart := 276410 },
  { event := event276431
    frameStart := 276410 }
]

def eventLeaf17277 : Array AnnotatedEvent := #[
  { event := event276432
    frameStart := 276410 },
  { event := event276433
    frameStart := 276410 },
  { event := event276434
    frameStart := 276410 },
  { event := event276435
    frameStart := 276410 },
  { event := event276436
    frameStart := 276410 },
  { event := event276437
    frameStart := 276410 },
  { event := event276438
    frameStart := 276410 },
  { event := event276439
    frameStart := 276410 },
  { event := event276440
    frameStart := 276410 },
  { event := event276441
    frameStart := 276410 },
  { event := event276442
    frameStart := 276410 },
  { event := event276443
    frameStart := 276410 },
  { event := event276444
    frameStart := 276410 },
  { event := event276445
    frameStart := 276410 },
  { event := event276446
    frameStart := 276410 },
  { event := event276447
    frameStart := 276410 }
]

def eventLeaf17278 : Array AnnotatedEvent := #[
  { event := event276448
    frameStart := 276410 },
  { event := event276449
    frameStart := 276410 },
  { event := event276450
    frameStart := 276410 },
  { event := event276451
    frameStart := 276410 },
  { event := event276452
    frameStart := 276410 },
  { event := event276453
    frameStart := 276410 },
  { event := event276454
    frameStart := 276410 },
  { event := event276455
    frameStart := 276410 },
  { event := event276456
    frameStart := 276410 },
  { event := event276457
    frameStart := 276410 },
  { event := event276458
    frameStart := 276410 },
  { event := event276459
    frameStart := 276410 },
  { event := event276460
    frameStart := 276410 },
  { event := event276461
    frameStart := 276410 },
  { event := event276462
    frameStart := 276410 },
  { event := event276463
    frameStart := 276410 }
]

def eventLeaf17279 : Array AnnotatedEvent := #[
  { event := event276464
    frameStart := 276410 },
  { event := event276465
    frameStart := 276410 },
  { event := event276466
    frameStart := 276410 },
  { event := event276467
    frameStart := 276410 },
  { event := event276468
    frameStart := 276410 },
  { event := event276469
    frameStart := 276410 },
  { event := event276470
    frameStart := 276410 },
  { event := event276471
    frameStart := 276410 },
  { event := event276472
    frameStart := 276410 },
  { event := event276473
    frameStart := 276410 },
  { event := event276474
    frameStart := 276410 },
  { event := event276475
    frameStart := 276410 },
  { event := event276476
    frameStart := 276410 },
  { event := event276477
    frameStart := 276410 },
  { event := event276478
    frameStart := 276410 },
  { event := event276479
    frameStart := 276410 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1079
