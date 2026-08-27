import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Replay420

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open Mxx.Certificate.OperationalNoise.TallSecurity0Generated

def replayState26880 : ReplayState := ⟨107520, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26881 : ReplayState := ⟨107524, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26882 : ReplayState := ⟨107528, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26883 : ReplayState := ⟨107532, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26884 : ReplayState := ⟨107536, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26885 : ReplayState := ⟨107540, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26886 : ReplayState := ⟨107544, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26887 : ReplayState := ⟨107548, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26888 : ReplayState := ⟨107552, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26889 : ReplayState := ⟨107556, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26890 : ReplayState := ⟨107560, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26891 : ReplayState := ⟨107564, [⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]⟩
def replayState26892 : ReplayState := ⟨107567, []⟩

theorem replayChunk26880 : ReplayChain document history replayState26880 replayState26881 :=
  .chunk 107524 (by rfl)

theorem replayChunk26881 : ReplayChain document history replayState26881 replayState26882 :=
  .chunk 107528 (by rfl)

theorem replayChunk26882 : ReplayChain document history replayState26882 replayState26883 :=
  .chunk 107532 (by rfl)

theorem replayChunk26883 : ReplayChain document history replayState26883 replayState26884 :=
  .chunk 107536 (by rfl)

theorem replayChunk26884 : ReplayChain document history replayState26884 replayState26885 :=
  .chunk 107540 (by rfl)

theorem replayChunk26885 : ReplayChain document history replayState26885 replayState26886 :=
  .chunk 107544 (by rfl)

theorem replayChunk26886 : ReplayChain document history replayState26886 replayState26887 :=
  .chunk 107548 (by rfl)

theorem replayChunk26887 : ReplayChain document history replayState26887 replayState26888 :=
  .chunk 107552 (by rfl)

theorem replayChunk26888 : ReplayChain document history replayState26888 replayState26889 :=
  .chunk 107556 (by rfl)

theorem replayChunk26889 : ReplayChain document history replayState26889 replayState26890 :=
  .chunk 107560 (by rfl)

theorem replayChunk26890 : ReplayChain document history replayState26890 replayState26891 :=
  .chunk 107564 (by rfl)

theorem replayChunk26891 : ReplayChain document history replayState26891 replayState26892 :=
  .chunk 107567 (by rfl)

theorem replayShard420 : ReplayChain document history replayState26880 replayState26892 :=
  (.trans (.trans (.trans replayChunk26880 (.trans replayChunk26881 replayChunk26882)) (.trans replayChunk26883 (.trans replayChunk26884 replayChunk26885))) (.trans (.trans replayChunk26886 (.trans replayChunk26887 replayChunk26888)) (.trans replayChunk26889 (.trans replayChunk26890 replayChunk26891))))

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Replay420
