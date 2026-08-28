import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard267
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard332
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard334
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard336

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult46232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46232
end ResidualResult46232

namespace ResidualResult46237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46005.actual selector witness *
    ResidualResult46232.actual selector witness
end ResidualResult46237

namespace ResidualResult46240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46240
end ResidualResult46240

namespace ResidualResult46244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46240.actual selector witness -
    ResidualResult46237.actual selector witness
end ResidualResult46244

namespace ResidualResult46248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46244.actual selector witness -
    ResidualResult46229.actual selector witness
end ResidualResult46248

namespace ResidualResult46291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult44882.actual selector witness
end ResidualResult46291

namespace ResidualResult46332
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46291.actual selector witness +
    ResidualResult44875.actual selector witness
end ResidualResult46332

namespace ResidualResult46342
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult46332.actual selector witness *
    ResidualResult5499.actual selector witness
end ResidualResult46342

namespace ResidualResult46346
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46346
end ResidualResult46346

namespace ResidualResult46349
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46349
end ResidualResult46349

namespace ResidualResult46359
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36323.actual selector witness *
    ResidualResult46349.actual selector witness
end ResidualResult46359

namespace ResidualResult46362
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46362
end ResidualResult46362

namespace ResidualResult46366
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46366
end ResidualResult46366

namespace ResidualResult46464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46464
end ResidualResult46464

namespace ResidualResult46475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46475
end ResidualResult46475

namespace ResidualResult46478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 46478
end ResidualResult46478

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
