import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard186
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard187

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult24543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24538.actual selector witness *
    ResidualResult24535.actual selector witness
end ResidualResult24543

namespace ResidualResult24547
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24543.actual selector witness -
    ResidualResult24520.actual selector witness
end ResidualResult24547

namespace ResidualResult24555
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24547.actual selector witness *
    ResidualResult24504.actual selector witness
end ResidualResult24555

namespace ResidualResult24558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24558
end ResidualResult24558

namespace ResidualResult24563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24515.actual selector witness *
    ResidualResult24558.actual selector witness
end ResidualResult24563

namespace ResidualResult24566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24566
end ResidualResult24566

namespace ResidualResult24570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24566.actual selector witness -
    ResidualResult24563.actual selector witness
end ResidualResult24570

namespace ResidualResult24574
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24570.actual selector witness -
    ResidualResult24555.actual selector witness
end ResidualResult24574

namespace ResidualResult24583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult24404.actual selector witness
end ResidualResult24583

namespace ResidualResult24590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24583.actual selector witness +
    ResidualResult24397.actual selector witness
end ResidualResult24590

namespace ResidualResult24600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult24590.actual selector witness *
    ResidualResult24313.actual selector witness
end ResidualResult24600

namespace ResidualResult24603
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24603
end ResidualResult24603

namespace ResidualResult24607
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24607
end ResidualResult24607

namespace ResidualResult24705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24705
end ResidualResult24705

namespace ResidualResult24716
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24716
end ResidualResult24716

namespace ResidualResult24719
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 24719
end ResidualResult24719

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
