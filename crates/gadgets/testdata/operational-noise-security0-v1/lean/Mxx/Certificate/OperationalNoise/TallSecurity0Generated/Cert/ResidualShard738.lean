import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard736
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard737

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult103554
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103550.actual selector witness +
    ResidualResult103491.actual selector witness
end ResidualResult103554

namespace ResidualResult103558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103554.actual selector witness +
    ResidualResult103488.actual selector witness
end ResidualResult103558

namespace ResidualResult103562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103558.actual selector witness +
    ResidualResult103485.actual selector witness
end ResidualResult103562

namespace ResidualResult103566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103562.actual selector witness +
    ResidualResult103482.actual selector witness
end ResidualResult103566

namespace ResidualResult103570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103566.actual selector witness +
    ResidualResult103479.actual selector witness
end ResidualResult103570

namespace ResidualResult103574
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103570.actual selector witness +
    ResidualResult103476.actual selector witness
end ResidualResult103574

namespace ResidualResult103578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103574.actual selector witness +
    ResidualResult103473.actual selector witness
end ResidualResult103578

namespace ResidualResult103582
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103578.actual selector witness +
    ResidualResult103470.actual selector witness
end ResidualResult103582

namespace ResidualResult103586
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103582.actual selector witness +
    ResidualResult103467.actual selector witness
end ResidualResult103586

namespace ResidualResult103590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103586.actual selector witness -
    ResidualResult103464.actual selector witness
end ResidualResult103590

namespace ResidualResult103666
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103590.actual selector witness *
    ResidualResult103431.actual selector witness
end ResidualResult103666

namespace ResidualResult103669
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 103669
end ResidualResult103669

namespace ResidualResult103674
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103442.actual selector witness *
    ResidualResult103669.actual selector witness
end ResidualResult103674

namespace ResidualResult103677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 103677
end ResidualResult103677

namespace ResidualResult103681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103677.actual selector witness -
    ResidualResult103674.actual selector witness
end ResidualResult103681

namespace ResidualResult103685
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult103681.actual selector witness -
    ResidualResult103666.actual selector witness
end ResidualResult103685

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
