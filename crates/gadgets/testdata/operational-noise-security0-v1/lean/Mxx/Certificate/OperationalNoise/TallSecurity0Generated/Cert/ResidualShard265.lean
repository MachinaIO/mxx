import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult36078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1593.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult36078

namespace ResidualResult36083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult6498.actual selector witness
end ResidualResult36083

namespace ResidualResult36087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36083.actual selector witness -
    ResidualResult36078.actual selector witness
end ResidualResult36087

namespace ResidualResult36093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36087.actual selector witness +
    ResidualResult6490.actual selector witness
end ResidualResult36093

namespace ResidualResult36103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36093.actual selector witness *
    ResidualResult6487.actual selector witness
end ResidualResult36103

namespace ResidualResult36109
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36103.actual selector witness +
    ResidualResult36073.actual selector witness
end ResidualResult36109

namespace ResidualResult36119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36109.actual selector witness *
    ResidualResult36040.actual selector witness
end ResidualResult36119

namespace ResidualResult36122
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36122
end ResidualResult36122

namespace ResidualResult36126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36126
end ResidualResult36126

namespace ResidualResult36131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult6550.actual selector witness
end ResidualResult36131

namespace ResidualResult36137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36131.actual selector witness +
    ResidualResult6548.actual selector witness
end ResidualResult36137

namespace ResidualResult36215
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36215
end ResidualResult36215

namespace ResidualResult36218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36218
end ResidualResult36218

namespace ResidualResult36223
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36218.actual selector witness *
    ResidualResult36215.actual selector witness
end ResidualResult36223

namespace ResidualResult36234
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36234
end ResidualResult36234

namespace ResidualResult36237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 36237
end ResidualResult36237

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
