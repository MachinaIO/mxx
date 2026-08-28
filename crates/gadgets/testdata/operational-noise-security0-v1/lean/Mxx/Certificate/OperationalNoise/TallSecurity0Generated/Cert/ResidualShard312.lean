import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard311

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult42267
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42259.actual selector witness *
    ResidualResult42236.actual selector witness
end ResidualResult42267

namespace ResidualResult42270
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42270
end ResidualResult42270

namespace ResidualResult42275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42247.actual selector witness *
    ResidualResult42270.actual selector witness
end ResidualResult42275

namespace ResidualResult42278
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42278
end ResidualResult42278

namespace ResidualResult42282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42278.actual selector witness -
    ResidualResult42275.actual selector witness
end ResidualResult42282

namespace ResidualResult42286
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42282.actual selector witness -
    ResidualResult42267.actual selector witness
end ResidualResult42286

namespace ResidualResult42295
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult42124.actual selector witness
end ResidualResult42295

namespace ResidualResult42302
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42295.actual selector witness +
    ResidualResult42117.actual selector witness
end ResidualResult42302

namespace ResidualResult42309
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42309
end ResidualResult42309

namespace ResidualResult42312
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42312
end ResidualResult42312

namespace ResidualResult42319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42319
end ResidualResult42319

namespace ResidualResult42322
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 42322
end ResidualResult42322

namespace ResidualResult42327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1889.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult42327

namespace ResidualResult42332
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult12985.actual selector witness
end ResidualResult42332

namespace ResidualResult42336
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42332.actual selector witness -
    ResidualResult42327.actual selector witness
end ResidualResult42336

namespace ResidualResult42342
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult42336.actual selector witness +
    ResidualResult12977.actual selector witness
end ResidualResult42342

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
