import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard408

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult56418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56390.actual selector witness *
    ResidualResult56413.actual selector witness
end ResidualResult56418

namespace ResidualResult56421
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56421
end ResidualResult56421

namespace ResidualResult56425
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56421.actual selector witness -
    ResidualResult56418.actual selector witness
end ResidualResult56425

namespace ResidualResult56429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56425.actual selector witness -
    ResidualResult56410.actual selector witness
end ResidualResult56429

namespace ResidualResult56438
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult56267.actual selector witness
end ResidualResult56438

namespace ResidualResult56445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56438.actual selector witness +
    ResidualResult56260.actual selector witness
end ResidualResult56445

namespace ResidualResult56452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56452
end ResidualResult56452

namespace ResidualResult56455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56455
end ResidualResult56455

namespace ResidualResult56462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56462
end ResidualResult56462

namespace ResidualResult56465
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56465
end ResidualResult56465

namespace ResidualResult56470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2614.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult56470

namespace ResidualResult56475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult12484.actual selector witness
end ResidualResult56475

namespace ResidualResult56479
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56475.actual selector witness -
    ResidualResult56470.actual selector witness
end ResidualResult56479

namespace ResidualResult56485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56479.actual selector witness +
    ResidualResult12476.actual selector witness
end ResidualResult56485

namespace ResidualResult56493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56485.actual selector witness *
    ResidualResult2617.actual selector witness
end ResidualResult56493

namespace ResidualResult56498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2617.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult56498

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
