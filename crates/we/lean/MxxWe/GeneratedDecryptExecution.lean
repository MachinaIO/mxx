import MxxWe.GenericDiamondExecution
import MxxWe.Generated.DiamondWeFamily.Statement

open Mxx
open MxxWe.Generated.DiamondWeFamily

namespace MxxWe

/-! Inversion helpers for the generated Diamond decryption stage. -/

/-- Extract one selected root-node evaluation from an arbitrary decryption-stage outcome.  The
generic path splitter avoids destructing every preceding generated node. -/
theorem decryptNode_of_member
    (samplers : MxxSamplerFamily)
    (p : DiamondWeFamilyParams)
    (inputs output : Mxx.Ir.Environment)
    (preNodes postNodes : List Mxx.Ir.Node)
    (node : Mxx.Ir.Node)
    (rootNodes : DiamondWeFamily_stage_decrypt.root.nodes =
      preNodes ++ node :: postNodes)
    (member : output ∈ Mxx.Ir.denote samplers DiamondWeFamily_stage_decrypt
      (DiamondWeFamilyParamEnvironment p) inputs) :
    ∃ before values,
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers DiamondWeFamily_stage_decrypt
          DiamondWeFamily_stage_decrypt.definitions.length)
        samplers (DiamondWeFamilyParamEnvironment p) inputs before node := by
  unfold Mxx.Ir.denote at member
  rw [Mxx.Ir.denoteScopeWithFuel_succ] at member
  simp only [List.mem_map] at member
  obtain ⟨wires, wiresMember, _⟩ := member
  obtain ⟨initial, initialMember, path⟩ :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).mp wiresMember
  simp only [List.mem_singleton] at initialMember
  subst initial
  rw [rootNodes] at path
  obtain ⟨before, values, _, valuesMember, _⟩ := path.atNode
  exact ⟨before, values, valuesMember⟩

/-- Index-based form used for generated roots. -/
theorem decryptNodeAt_of_member
    (samplers : MxxSamplerFamily)
    (p : DiamondWeFamilyParams)
    (inputs output : Mxx.Ir.Environment)
    (index : Nat)
    (indexValid : index < DiamondWeFamily_stage_decrypt.root.nodes.length)
    (member : output ∈ Mxx.Ir.denote samplers DiamondWeFamily_stage_decrypt
      (DiamondWeFamilyParamEnvironment p) inputs) :
    ∃ before values,
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers DiamondWeFamily_stage_decrypt
          DiamondWeFamily_stage_decrypt.definitions.length)
        samplers (DiamondWeFamilyParamEnvironment p) inputs before
          DiamondWeFamily_stage_decrypt.root.nodes[index] := by
  apply decryptNode_of_member samplers p inputs output
    (DiamondWeFamily_stage_decrypt.root.nodes.take index)
    (DiamondWeFamily_stage_decrypt.root.nodes.drop (index + 1))
    DiamondWeFamily_stage_decrypt.root.nodes[index]
  · calc
      DiamondWeFamily_stage_decrypt.root.nodes =
          DiamondWeFamily_stage_decrypt.root.nodes.take index ++
            DiamondWeFamily_stage_decrypt.root.nodes.drop index :=
        (List.take_append_drop index DiamondWeFamily_stage_decrypt.root.nodes).symm
      _ = DiamondWeFamily_stage_decrypt.root.nodes.take index ++
          DiamondWeFamily_stage_decrypt.root.nodes[index] ::
            DiamondWeFamily_stage_decrypt.root.nodes.drop (index + 1) := by
        rw [List.drop_eq_getElem_cons indexValid]
  · exact member

end MxxWe
