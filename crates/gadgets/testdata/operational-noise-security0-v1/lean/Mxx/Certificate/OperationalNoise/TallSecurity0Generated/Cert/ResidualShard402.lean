import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard097
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult55488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 55488
end ResidualResult55488

namespace ResidualResult55491
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 55491
end ResidualResult55491

namespace ResidualResult55498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 55498
end ResidualResult55498

namespace ResidualResult55501
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 55501
end ResidualResult55501

namespace ResidualResult55506
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2568.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult55506

namespace ResidualResult55511
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult11482.actual selector witness
end ResidualResult55511

namespace ResidualResult55515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55511.actual selector witness -
    ResidualResult55506.actual selector witness
end ResidualResult55515

namespace ResidualResult55521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55515.actual selector witness +
    ResidualResult11474.actual selector witness
end ResidualResult55521

namespace ResidualResult55529
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55521.actual selector witness *
    ResidualResult2571.actual selector witness
end ResidualResult55529

namespace ResidualResult55534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2571.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult55534

namespace ResidualResult55539
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult11523.actual selector witness
end ResidualResult55539

namespace ResidualResult55543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55539.actual selector witness -
    ResidualResult55534.actual selector witness
end ResidualResult55543

namespace ResidualResult55549
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55543.actual selector witness +
    ResidualResult11515.actual selector witness
end ResidualResult55549

namespace ResidualResult55559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55549.actual selector witness *
    ResidualResult11512.actual selector witness
end ResidualResult55559

namespace ResidualResult55565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55559.actual selector witness +
    ResidualResult55529.actual selector witness
end ResidualResult55565

namespace ResidualResult55575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55565.actual selector witness *
    ResidualResult55501.actual selector witness
end ResidualResult55575

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
