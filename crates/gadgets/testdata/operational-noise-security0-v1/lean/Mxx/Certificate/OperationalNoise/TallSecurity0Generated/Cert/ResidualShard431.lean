import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard401
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard405
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard409
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard413
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard416
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard420
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard424
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard427
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard430

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult59302
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59294.actual selector witness *
    ResidualResult59271.actual selector witness
end ResidualResult59302

namespace ResidualResult59305
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 59305
end ResidualResult59305

namespace ResidualResult59310
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59282.actual selector witness *
    ResidualResult59305.actual selector witness
end ResidualResult59310

namespace ResidualResult59313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 59313
end ResidualResult59313

namespace ResidualResult59317
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59313.actual selector witness -
    ResidualResult59310.actual selector witness
end ResidualResult59317

namespace ResidualResult59321
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59317.actual selector witness -
    ResidualResult59302.actual selector witness
end ResidualResult59321

namespace ResidualResult59330
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult59159.actual selector witness
end ResidualResult59330

namespace ResidualResult59337
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59330.actual selector witness +
    ResidualResult59152.actual selector witness
end ResidualResult59337

namespace ResidualResult59342
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59337.actual selector witness +
    ResidualResult58855.actual selector witness
end ResidualResult59342

namespace ResidualResult59347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59342.actual selector witness +
    ResidualResult58373.actual selector witness
end ResidualResult59347

namespace ResidualResult59352
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59347.actual selector witness +
    ResidualResult57891.actual selector witness
end ResidualResult59352

namespace ResidualResult59357
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59352.actual selector witness +
    ResidualResult57409.actual selector witness
end ResidualResult59357

namespace ResidualResult59362
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59357.actual selector witness +
    ResidualResult56927.actual selector witness
end ResidualResult59362

namespace ResidualResult59367
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59362.actual selector witness +
    ResidualResult56445.actual selector witness
end ResidualResult59367

namespace ResidualResult59372
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59367.actual selector witness +
    ResidualResult55963.actual selector witness
end ResidualResult59372

namespace ResidualResult59377
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult59372.actual selector witness +
    ResidualResult55481.actual selector witness
end ResidualResult59377

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
