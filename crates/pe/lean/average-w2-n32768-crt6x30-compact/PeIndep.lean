import PeLink

/-!
# Independence from the key extension

The samples of `prims` read only the tape keys of `preKeys`: the sites of the secret, TrapGen, the
prelude, and encryption. Key generation samples at sites under `[6]`, none of which is in
`preKeys`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The sample sites that precede key generation. -/
@[irreducible] def preSites : List (List Nat) :=
  [[2, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2], [3, 6], [4, 28], [4, 39], [4, 5, 0, 14],
    [4, 5, 0, 23]] ++ (List.range 24).map (fun j ↦ [3, 11, 0, 3, j, 2]) ++
  (List.range 576).flatMap fun i ↦ [[4, 46, i, 29], [4, 46, i, 39], [4, 8, i, 35], [4, 8, i, 45]]

/-- The laws of the samples that precede key generation. -/
@[irreducible] def preLaws : List SamplerLaw :=
  [.interval (-1) 1, .residue Q, .gaussian trapSigma trapGenCutoff, preludeLaw, encLaw, helperLaw]

/-- The tape keys that precede key generation. -/
@[irreducible] noncomputable def preKeys : Finset SampleKey :=
  (preSites.toFinset ×ˢ (Finset.range 14 ×ˢ (Finset.range 24 ×ˢ (Finset.range N ×ˢ preLaws.toFinset)))).image
    fun x ↦ ⟨x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2⟩

theorem mem_preKeys {site : List Nat} {r c k : Nat} {law : SamplerLaw} (hs : site ∈ preSites)
    (hr : r < 14) (hc : c < 24) (hk : k < N) (hl : law ∈ preLaws) :
    (⟨site, r, c, k, law⟩ : SampleKey) ∈ preKeys := by
  unfold preKeys
  refine Finset.mem_image.mpr ⟨(site, r, c, k, law), ?_, ?_⟩
  · rw [Finset.mem_product, Finset.mem_product, Finset.mem_product, Finset.mem_product, List.mem_toFinset,
      List.mem_toFinset, Finset.mem_range, Finset.mem_range, Finset.mem_range]
    exact ⟨hs, hr, hc, hk, hl⟩
  · dsimp only

theorem lane_site_mem {i : Nat} (hi : i < 576) :
    [4, 46, i, 29] ∈ preSites ∧ [4, 46, i, 39] ∈ preSites ∧ [4, 8, i, 35] ∈ preSites ∧
      [4, 8, i, 45] ∈ preSites := by
  have h : ∀ s ∈ [[4, 46, i, 29], [4, 46, i, 39], [4, 8, i, 35], [4, 8, i, 45]], s ∈ preSites := by
    intro s hs
    unfold preSites
    exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨i, List.mem_range.mpr hi, hs⟩)
  exact ⟨h _ (by simp), h _ (by simp), h _ (by simp), h _ (by simp)⟩

theorem z_site_mem {j : Nat} (hj : j < 24) : [3, 11, 0, 3, j, 2] ∈ preSites := by
  unfold preSites
  exact List.mem_append_left _ (List.mem_append_right _ (List.mem_map.mpr ⟨j, List.mem_range.mpr hj, rfl⟩))

/-- Every site of `preSites` starts with `0`, `2`, `3`, or `4`. -/
theorem preSites_head {s : List Nat} (hs : s ∈ preSites) : s.head? ≠ some 6 := by
  simp only [preSites, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range] at hs
  rcases hs with (hs | ⟨j, -, rfl⟩) | ⟨i, -, hs⟩
  · simp at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
  · simp
  · simp at hs
    rcases hs with rfl | rfl | rfl | rfl <;> simp

theorem not_mem_preKeys {site : List Nat} (hs : site.head? = some 6) {r c k : Nat} {law : SamplerLaw} :
    (⟨site, r, c, k, law⟩ : SampleKey) ∉ preKeys := by
  intro h
  unfold preKeys at h
  obtain ⟨x, hx, he⟩ := Finset.mem_image.mp h
  simp only [Finset.mem_product, List.mem_toFinset] at hx
  have hsite : x.1 = site := congrArg SampleKey.site he
  exact preSites_head (hsite ▸ hx.1) hs

theorem tapeWitness_congr {rows columns : Nat} {ω ω' : SampleTape} {site : List Nat} {law : SamplerLaw}
    (h : ∀ (r : Fin rows) (c : Fin columns) (k : Fin N),
      ω ⟨site, r.val, c.val, k.val, law⟩ = ω' ⟨site, r.val, c.val, k.val, law⟩) :
    (tapeWitness ω site law : ErrorMatrix N rows columns) = tapeWitness ω' site law := by
  funext r c
  exact congrArg intPoly (funext fun k ↦ h r c k)

theorem preLaws_mem : SamplerLaw.interval (-1) 1 ∈ preLaws ∧ SamplerLaw.residue Q ∈ preLaws ∧
    SamplerLaw.gaussian trapSigma trapGenCutoff ∈ preLaws ∧ preludeLaw ∈ preLaws ∧ encLaw ∈ preLaws ∧
    helperLaw ∈ preLaws := by
  unfold preLaws
  simp

theorem fixed_site_mem {s : List Nat}
    (h : s ∈ [[2, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2], [3, 6], [4, 28], [4, 39],
      [4, 5, 0, 14], [4, 5, 0, 23]]) : s ∈ preSites := by
  unfold preSites
  exact List.mem_append_left _ (List.mem_append_left _ h)

/-- Tapes that agree on `preKeys` have the same samples before key generation. -/
theorem prims_congr {ω ω' : SampleTape} (h : ∀ k ∈ preKeys, ω k = ω' k) : prims ω = prims ω' := by
  obtain ⟨l1, l2, l3, l4, l5, l6⟩ := preLaws_mem
  have hw : ∀ {rows columns : Nat} {site : List Nat} {law : SamplerLaw}, site ∈ preSites →
      law ∈ preLaws → rows ≤ 14 → columns ≤ 24 →
      (tapeWitness ω site law : ErrorMatrix N rows columns) = tapeWitness ω' site law := by
    intro rows columns site law hs hl hr hc
    exact tapeWitness_congr fun r c k ↦ h _ (mem_preKeys hs (by omega) (by omega) k.isLt hl)
  have hm : ∀ {rows columns : Nat} {site : List Nat} {law : SamplerLaw}, site ∈ preSites →
      law ∈ preLaws → rows ≤ 14 → columns ≤ 24 →
      (tapeMatrix ω site law : ExactMatrix Q N rows columns) = tapeMatrix ω' site law := by
    intro rows columns site law hs hl hr hc
    exact tapeMatrix_congr fun r c k ↦ h _ (mem_preKeys hs (by omega) (by omega) k.isLt hl)
  have hsec : secOf ω = secOf ω' := by
    unfold secOf
    rw [hw (rows := 1) (columns := 1) (fixed_site_mem (by simp)) l1 (by omega) (by omega)]
  have hB : BOf ω = BOf ω' := by
    unfold BOf
    rw [hm (rows := 1) (columns := 1) (site := [0, 0, 2]) (fixed_site_mem (by simp)) l2 (by omega) (by omega),
      hm (rows := 1) (columns := 1 * Backend.layout0.digitCount) (site := [0, 0, 0])
        (fixed_site_mem (by simp)) l3 (by omega) (by decide),
      hm (rows := 1) (columns := 1 * Backend.layout0.digitCount) (site := [0, 0, 1])
        (fixed_site_mem (by simp)) l3 (by omega) (by decide)]
  have heb : ebAt ω [3] = ebAt ω' [3] :=
    hw (site := [3, 6]) (fixed_site_mem (by simp)) l4 (by omega) (by omega)
  have hZ : zAt ω [3] = zAt ω' [3] := by
    funext r c
    unfold zAt zColAt
    exact congrFun (congrFun (hw (rows := 14) (columns := 1) (site := [3] ++ [11, 0, 3, (c : Nat), 2])
      (by simpa using z_site_mem c.isLt) l4 (by omega) (by omega)) r) 0
  have he0 : e0Of ω = e0Of ω' := by
    unfold e0Of errAt
    rw [hw (site := [4, 28]) (fixed_site_mem (by simp)) l5 (by omega) (by omega),
      hw (site := [4, 39]) (fixed_site_mem (by simp)) l5 (by omega) (by omega)]
  have heh : ehOf ω = ehOf ω' := by
    unfold ehOf errAt
    rw [hw (site := [4, 5, 0, 14]) (fixed_site_mem (by simp)) l6 (by omega) (by omega),
      hw (site := [4, 5, 0, 23]) (fixed_site_mem (by simp)) l6 (by omega) (by omega)]
  have hez : ezOf ω = ezOf ω' := by
    funext jj a
    have hi : (jj : Nat) + 24 * (a : Nat) < 576 := by omega
    obtain ⟨h1, h2, -, -⟩ := lane_site_mem hi
    unfold ezOf errAt
    rw [hw h1 l5 (by omega) (by omega), hw h2 l5 (by omega) (by omega)]
  have heza : ezaOf ω = ezaOf ω' := by
    funext jj a
    have hi : (jj : Nat) + 24 * (a : Nat) < 576 := by omega
    obtain ⟨-, -, h3, h4⟩ := lane_site_mem hi
    unfold ezaOf errAt
    rw [hw h3 l5 (by omega) (by omega), hw h4 l5 (by omega) (by omega)]
  unfold prims
  rw [hsec, hB, heb, hZ, he0, heh, hez, heza]

end PeProof
