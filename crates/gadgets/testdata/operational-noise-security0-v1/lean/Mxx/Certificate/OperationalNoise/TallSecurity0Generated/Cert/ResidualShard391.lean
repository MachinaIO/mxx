import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult54045
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 54045
end ResidualResult54045

namespace ResidualResult54052
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 54052
end ResidualResult54052

namespace ResidualResult54055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 54055
end ResidualResult54055

namespace ResidualResult54060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2499.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult54060

namespace ResidualResult54065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult9979.actual selector witness
end ResidualResult54065

namespace ResidualResult54069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54065.actual selector witness -
    ResidualResult54060.actual selector witness
end ResidualResult54069

namespace ResidualResult54075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54069.actual selector witness +
    ResidualResult9971.actual selector witness
end ResidualResult54075

namespace ResidualResult54083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54075.actual selector witness *
    ResidualResult2502.actual selector witness
end ResidualResult54083

namespace ResidualResult54088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2502.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult54088

namespace ResidualResult54093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult10020.actual selector witness
end ResidualResult54093

namespace ResidualResult54097
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54093.actual selector witness -
    ResidualResult54088.actual selector witness
end ResidualResult54097

namespace ResidualResult54103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54097.actual selector witness +
    ResidualResult10012.actual selector witness
end ResidualResult54103

namespace ResidualResult54113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54103.actual selector witness *
    ResidualResult10009.actual selector witness
end ResidualResult54113

namespace ResidualResult54119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54113.actual selector witness +
    ResidualResult54083.actual selector witness
end ResidualResult54119

namespace ResidualResult54129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult54119.actual selector witness *
    ResidualResult54055.actual selector witness
end ResidualResult54129

namespace ResidualResult54132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 54132
end ResidualResult54132

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
