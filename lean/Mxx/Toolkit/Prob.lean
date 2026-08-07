import Mxx.Correctness

namespace Mxx.Toolkit

def support {α : Type} (distribution : List α) : Set α :=
  { value | value ∈ distribution }

def equalFailure {α : Type} [DecidableEq α] (concrete ideal : α) : Bool :=
  decide (concrete ≠ ideal)

def equalAfterMapFailure {α β : Type} [DecidableEq β]
    (map : α → Option β) (concrete : α) (ideal : β) : Bool :=
  match map concrete with
  | none => true
  | some mapped => decide (mapped ≠ ideal)

def normWithinFailure {α : Type}
    (distance : α → α → Option Nat) (bound : Nat) (concrete ideal : α) : Bool :=
  match distance concrete ideal with
  | none => true
  | some actual => decide (actual > bound)

def supportCorrect {α : Type} (distribution : List α) (bad : α → Prop) : Prop :=
  ∀ value ∈ distribution, ¬ bad value

theorem failureProbability_eq_zero_of_support
    {α : Type} (distribution : List α) (bad : α → Prop) [DecidablePred bad]
    (safe : ∀ value ∈ distribution, ¬ bad value) :
    (if distribution.any (fun value => decide (bad value)) then (1 : ENNReal) else 0) = 0 := by
  split
  · rename_i found
    simp only [List.any_eq_true] at found
    obtain ⟨value, member, bad_value⟩ := found
    exact (safe value member (of_decide_eq_true bad_value)).elim
  · rfl

theorem supportCorrect.failureProbability_eq_zero
    {α : Type} (distribution : List α) (bad : α → Prop) [DecidablePred bad]
    (safe : supportCorrect distribution bad) :
    (if distribution.any (fun value => decide (bad value)) then (1 : ENNReal) else 0) = 0 :=
  failureProbability_eq_zero_of_support distribution bad safe

end Mxx.Toolkit
