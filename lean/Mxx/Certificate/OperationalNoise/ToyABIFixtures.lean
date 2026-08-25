import Mxx.Certificate.OperationalNoise.ToyABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

def fixtureSource : ToySource := expectedSource
def fixtureDocument : SchemaV1.Document := expectedDocument
def fixtureRows : ToyRows := expectedRows
def fixtureEvents : List ToyEvent := expectedEvents

theorem fixture_valid : ToyValid fixtureSource fixtureDocument fixtureRows fixtureEvents := by
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  intro index indexBound
  rfl

def honestEnv (gaussian : Int) : ToyEnv := fun owner =>
  if owner = o 0 then 258
  else if owner = o 7 ∨ owner = o 1 then 1
  else if owner = o 3 then gaussian else 0

theorem honestUniversal (gaussian : Int) :
    (m [0, 7]).eval (honestEnv gaussian) % Int.ofNat 257 =
      (m [1]).eval (honestEnv gaussian) % Int.ofNat 257 := by
  simp [m, ToyMonomial.eval, ToyMonomial.toSemanticKey, TallSemantics.evalMonomial,
    honestEnv, o]

theorem honestUniversal_not_exact (gaussian : Int) :
    (m [0, 7]).eval (honestEnv gaussian) ≠ (m [1]).eval (honestEnv gaussian) := by
  simp [m, ToyMonomial.eval, ToyMonomial.toSemanticKey, TallSemantics.evalMonomial,
    honestEnv, o]

def honestWitness (gaussian : Int) (bound : centeredNorm 257 gaussian ≤ 1) :
    ToyReplayWitness fixtureEvents where
  env := honestEnv gaussian
  gaussianEvent := by decide
  gaussianBound := by simpa [honestEnv, o] using bound
  universalEvent := by decide
  universalRelation := honestUniversal gaussian

theorem fixture_negative_gaussian :
    ToyOperationalClaim fixtureEvents (honestWitness (-1) (by decide)) :=
  operationalProof fixture_valid _

theorem fixture_positive_gaussian :
    ToyOperationalClaim fixtureEvents (honestWitness 1 (by decide)) :=
  operationalProof fixture_valid _

theorem toy_event_replay :
    ToyValid fixtureSource fixtureDocument fixtureRows fixtureEvents ∧
      (m [0, 7]).eval (honestEnv (-1)) ≠ (m [1]).eval (honestEnv (-1)) ∧
      ToyOperationalClaim fixtureEvents (honestWitness (-1) (by decide)) ∧
      ToyOperationalClaim fixtureEvents (honestWitness 1 (by decide)) :=
  ⟨fixture_valid, honestUniversal_not_exact (-1), fixture_negative_gaussian,
    fixture_positive_gaussian⟩

#print axioms toy_event_replay
#print axioms replay_sound
#print axioms operationalProof

end Mxx.Certificate.OperationalNoise.ToyABI
