import PePreGlue

/-!
# The grouped preprocessing leaf artifact layout

There are four groups, each with fifteen member lanes and four local slots. The flattened
slot `4 * group + localSlot` is zero for the first-depth leaf; flattened slot `1 + parent`
contains second-depth leaf `parent + 15 * member`. Thus leaf `l` uses group
`(1 + l % 15) / 4`, local slot `(1 + l % 15) % 4`, and member `l / 15`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- View the flattened group/slot/member families as the second-depth leaf family. -/
def groupedPreLeafRows
    (rows : Fin 16 → Fin 2 → Fin 15 → ExactMatrix Q N 1 60)
    (r : Fin 2) (leaf : Fin 225) : ExactMatrix Q N 1 60 :=
  rows ⟨1 + leaf.val % 15, by omega⟩ r ⟨leaf.val / 15, by omega⟩

/-- The inverse layout map used when summing a member's fifteen second-depth slots. -/
theorem groupedPreLeafRows_reindex
    (rows : Fin 16 → Fin 2 → Fin 15 → ExactMatrix Q N 1 60)
    (parent : Fin 15) (member : Fin 15) (r : Fin 2) :
    groupedPreLeafRows rows r ⟨parent.val + 15 * member.val, by omega⟩ =
      rows parent.succ r member := by
  unfold groupedPreLeafRows
  congr 1
  · apply Fin.ext
    simp only [Fin.val_succ]
    omega
  · apply Fin.ext
    simp only
    omega

/-- A grouped slot has exactly the old second-depth switched key. -/
theorem groupedPreLeaf_key2 {model : HashModel} {key : ByteArray}
    (rows : Fin 16 → Fin 2 → Fin 15 → ExactMatrix Q N 1 60)
    (data : Leaf2Data) (hk : data.key = key)
    (h0 : data.T0 = groupedPreLeafRows rows 0)
    (h1 : data.T1 = groupedPreLeafRows rows 1)
    (parent : Fin 15) (member : Fin 15) (r j : Fin 2) :
    pairMul (helperKey model key (kSw (15 + member)) r)
      (actionDigits (rows parent.succ 0 member) (rows parent.succ 1 member) j) =
      data.key2 model ⟨parent.val + 15 * member.val, by omega⟩ r j := by
  rw [key2_eq data hk h0 h1]
  have hmember : (parent.val + 15 * member.val) / 15 = member.val := by omega
  simp only [Fin.val_mk]
  rw [hmember, groupedPreLeafRows_reindex, groupedPreLeafRows_reindex]

end PeProof
