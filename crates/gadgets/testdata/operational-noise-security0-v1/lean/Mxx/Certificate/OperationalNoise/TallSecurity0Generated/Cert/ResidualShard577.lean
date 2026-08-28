import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard576

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult81327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81327
end ResidualResult81327

namespace ResidualResult81331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81327.actual selector witness -
    ResidualResult81324.actual selector witness
end ResidualResult81331

namespace ResidualResult81335
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81331.actual selector witness -
    ResidualResult81316.actual selector witness
end ResidualResult81335

namespace ResidualResult81344
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult81173.actual selector witness
end ResidualResult81344

namespace ResidualResult81351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81344.actual selector witness +
    ResidualResult81166.actual selector witness
end ResidualResult81351

namespace ResidualResult81358
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81358
end ResidualResult81358

namespace ResidualResult81361
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81361
end ResidualResult81361

namespace ResidualResult81368
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81368
end ResidualResult81368

namespace ResidualResult81371
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81371
end ResidualResult81371

namespace ResidualResult81376
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3897.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult81376

namespace ResidualResult81381
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult7975.actual selector witness
end ResidualResult81381

namespace ResidualResult81385
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81381.actual selector witness -
    ResidualResult81376.actual selector witness
end ResidualResult81385

namespace ResidualResult81391
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81385.actual selector witness +
    ResidualResult7967.actual selector witness
end ResidualResult81391

namespace ResidualResult81399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81391.actual selector witness *
    ResidualResult3900.actual selector witness
end ResidualResult81399

namespace ResidualResult81404
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3900.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult81404

namespace ResidualResult81409
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult8016.actual selector witness
end ResidualResult81409

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
