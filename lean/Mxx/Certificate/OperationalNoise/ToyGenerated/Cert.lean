import Mxx.Certificate.OperationalNoise.ToyABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyGenerated

open Mxx.Certificate.OperationalNoise
open SchemaV1
open ToyABI

def source : ToySource :=
  { schemaId := "mxx.operational-noise.toy-source"
    schemaVersion := 1
    abi := "singleton-preimage-gaussian-v1"
    rustProjectionVersion := "operational-noise-certificate-v1"
    leanAbiVersion := "toy-replay-v1"
    request := ⟨"singleton-preimage-gaussian", [], []⟩
    parameters := ⟨"2", "257", 1, "1", "3", "4", "2", "8", "1", "1"⟩ }

def document : Document :=
  { schemaId := "mxx.operational-noise.certificate"
    schemaVersion := 1
    plaintextModulus := "2"
    ciphertextModulus := "257"
    ringDimension := 1
    expressions :=
      [ { descriptor := .operation (.event (.sampler ⟨0⟩)) (toyMatrix 1 4)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨1⟩)) (toyMatrix 1 1)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨2⟩)) (toyMatrix 4 1)
          inputs := []
          program := none },
        { descriptor := .operation (.event (.sampler ⟨3⟩)) (toyMatrix 1 1)
          inputs := []
          program := none },
        { descriptor := .operation (.stable (.trapdoor
            (.generate "trapdoor-sample" [4, 2] (some (⟨0⟩)) "value"))) (.trapdoor)
          inputs := []
          program := none },
        { descriptor := .source (.direct ⟨0⟩)
          inputs := []
          program := none },
        { descriptor := .source (.direct ⟨1⟩)
          inputs := []
          program := none },
        { descriptor := .operation (.stable .programCall) (toyMatrix 4 1)
          inputs := [⟨5⟩]
          program := some ⟨0⟩ },
        { descriptor := .operation (.stable (.matrix .multiply)) (toyMatrix 1 1)
          inputs := [⟨0⟩, ⟨7⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .scale)) (toyMatrix 4 1)
          inputs := [⟨7⟩, ⟨6⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .multiply)) (toyMatrix 1 1)
          inputs := [⟨0⟩, ⟨9⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .subtract)) (toyMatrix 1 1)
          inputs := [⟨10⟩, ⟨1⟩]
          program := none },
        { descriptor := .operation (.stable (.matrix .add)) (toyMatrix 1 1)
          inputs := [⟨11⟩, ⟨3⟩]
          program := none } ]
    programs := [⟨[⟨.int, some (⟨0, 1⟩)⟩],
        toyMatrix 4 1,
        some (⟨⟨0, 1⟩, toyMatrix 4 1, false, none⟩),
        ⟨2⟩⟩]
    sources := [.constant ⟨.int, .int "0"⟩, .constant ⟨.int, .int "1"⟩]
    events := [.sampler (toyWire 0)
        (.trapdoor (toyMatrix 1 4)
          "{\"tag\":\"Rational\",\"value\":{\"numerator\":\"3\",\"denominator\":\"1\"}}"
          4 2 "8") none,
      .sampler (toyWire 1)
        (.uniformResidue (toyMatrix 1 1)) none,
      .sampler (toyWire 2)
        (.preimage (toyMatrix 4 1) "8") none,
      .sampler (toyWire 6)
        (.gaussian (toyMatrix 1 1)
          "{\"tag\":\"Rational\",\"value\":{\"numerator\":\"1\",\"denominator\":\"1\"}}" "1") none]
    indexUses := []
    sliceGroups := []
    residualRoot := .closed ⟨12⟩ }

def rows : ToyRows :=
  { expressions := [⟨0⟩, ⟨1⟩, ⟨2⟩, ⟨3⟩, ⟨4⟩, ⟨5⟩, ⟨6⟩,
      ⟨7⟩, ⟨8⟩, ⟨9⟩, ⟨10⟩, ⟨11⟩, ⟨12⟩]
    program := ⟨0⟩
    sources := [⟨0⟩, ⟨1⟩]
    events := [⟨0⟩, ⟨1⟩, ⟨2⟩, ⟨3⟩]
    root := ⟨12⟩ }

end Mxx.Certificate.OperationalNoise.ToyGenerated
