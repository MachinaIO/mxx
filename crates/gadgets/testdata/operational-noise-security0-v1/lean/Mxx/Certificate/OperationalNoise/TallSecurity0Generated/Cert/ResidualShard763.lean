import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard762

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult107087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107087
end ResidualResult107087

namespace ResidualResult107096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107096
end ResidualResult107096

namespace ResidualResult107098
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107098
end ResidualResult107098

namespace ResidualResult107103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107098.actual selector witness *
    ResidualResult107096.actual selector witness
end ResidualResult107103

namespace ResidualResult107106
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107106
end ResidualResult107106

namespace ResidualResult107110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107106.actual selector witness -
    ResidualResult107103.actual selector witness
end ResidualResult107110

namespace ResidualResult107118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107110.actual selector witness *
    ResidualResult107087.actual selector witness
end ResidualResult107118

namespace ResidualResult107121
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107121
end ResidualResult107121

namespace ResidualResult107126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107098.actual selector witness *
    ResidualResult107121.actual selector witness
end ResidualResult107126

namespace ResidualResult107129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 107129
end ResidualResult107129

namespace ResidualResult107133
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107129.actual selector witness -
    ResidualResult107126.actual selector witness
end ResidualResult107133

namespace ResidualResult107137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107133.actual selector witness -
    ResidualResult107118.actual selector witness
end ResidualResult107137

namespace ResidualResult107146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult106999.actual selector witness
end ResidualResult107146

namespace ResidualResult107153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107146.actual selector witness +
    ResidualResult106992.actual selector witness
end ResidualResult107153

namespace ResidualResult107163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult107153.actual selector witness *
    ResidualResult5859.actual selector witness
end ResidualResult107163

namespace ResidualResult107168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult107168

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
