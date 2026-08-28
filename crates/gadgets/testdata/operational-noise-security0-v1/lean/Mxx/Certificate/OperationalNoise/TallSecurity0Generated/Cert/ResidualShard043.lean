import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard042

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult5276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5272.actual selector witness -
    ResidualResult5056.actual selector witness
end ResidualResult5276

namespace ResidualResult5299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5276.actual selector witness *
    ResidualResult4563.actual selector witness
end ResidualResult5299

namespace ResidualResult5303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult728.actual selector witness +
    ResidualResult5299.actual selector witness
end ResidualResult5303

namespace ResidualResult5307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5303.actual selector witness +
    ResidualResult4561.actual selector witness
end ResidualResult5307

namespace ResidualResult5311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5307.actual selector witness +
    ResidualResult3819.actual selector witness
end ResidualResult5311

namespace ResidualResult5315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5311.actual selector witness +
    ResidualResult3071.actual selector witness
end ResidualResult5315

namespace ResidualResult5319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5315.actual selector witness +
    ResidualResult2323.actual selector witness
end ResidualResult5319

namespace ResidualResult5323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5319.actual selector witness +
    ResidualResult1575.actual selector witness
end ResidualResult5323

namespace ResidualResult5327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5323.actual selector witness +
    ResidualResult827.actual selector witness
end ResidualResult5327

namespace ResidualResult5464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5327.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult5464

namespace ResidualResult5469
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2.actual selector witness *
    ResidualResult34.actual selector witness
end ResidualResult5469

namespace ResidualResult5472
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5472
end ResidualResult5472

namespace ResidualResult5476
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5476
end ResidualResult5476

namespace ResidualResult5480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5480
end ResidualResult5480

namespace ResidualResult5483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5483
end ResidualResult5483

namespace ResidualResult5487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5487
end ResidualResult5487

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
