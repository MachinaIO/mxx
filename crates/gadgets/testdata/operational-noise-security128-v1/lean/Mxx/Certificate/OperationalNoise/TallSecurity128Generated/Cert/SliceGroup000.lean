import Mxx.Certificate.OperationalNoise.CertificateABI
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.SliceGroupRows0Level2_0000

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.SliceGroup000

open Mxx.Certificate.OperationalNoise
open SchemaV1
open CertificateABI

def SliceGroupRow0 : SchemaV1.SliceGroupRow :=
  ⟨⟨"encoding", .parallelBody (.root) 2, 7, 1, 0⟩, some (.expression ⟨2373⟩), some (.expression ⟨0⟩), .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 1, [.argument (⟨.parallelBody (.root) 2, 7⟩) (.expression ⟨97⟩) 0 (0, 655360)], some (1), some (1), [⟨.rowStart, .expression ⟨97⟩, ⟨0, 655360⟩⟩, ⟨.rowEndExclusive, .expression ⟨2372⟩, ⟨1, 655361⟩⟩, ⟨.columnStart, .expression ⟨136⟩, ⟨0, 1⟩⟩, ⟨.columnEndExclusive, .expression ⟨2370⟩, ⟨1, 2⟩⟩], Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.SliceGroupRows0Level2_0000.rows⟩

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.SliceGroup000
