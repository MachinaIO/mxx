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
  if owner = o 0 ∨ owner = o 7 ∨ owner = o 1 then 1
  else if owner = o 3 then gaussian else 0

theorem honestUniversal (gaussian : Int) :
    (m [0, 7]).eval (honestEnv gaussian) % 257 = (m [1]).eval (honestEnv gaussian) % 257 := by
  simp [m, ToyMonomial.eval, honestEnv, o]

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
      ToyOperationalClaim fixtureEvents (honestWitness (-1) (by decide)) ∧
      ToyOperationalClaim fixtureEvents (honestWitness 1 (by decide)) :=
  ⟨fixture_valid, fixture_negative_gaussian, fixture_positive_gaussian⟩

#print axioms toy_event_replay
#print axioms replay_sound
#print axioms operationalProof

end Mxx.Certificate.OperationalNoise.ToyABI
