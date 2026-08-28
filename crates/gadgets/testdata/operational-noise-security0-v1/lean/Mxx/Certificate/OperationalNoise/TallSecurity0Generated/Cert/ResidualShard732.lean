import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard710
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard714
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard717
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard721
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard725
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard728
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard731

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult102126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 102126
end ResidualResult102126

namespace ResidualResult102130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102126.actual selector witness -
    ResidualResult102123.actual selector witness
end ResidualResult102130

namespace ResidualResult102138
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102130.actual selector witness *
    ResidualResult102107.actual selector witness
end ResidualResult102138

namespace ResidualResult102141
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 102141
end ResidualResult102141

namespace ResidualResult102146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102118.actual selector witness *
    ResidualResult102141.actual selector witness
end ResidualResult102146

namespace ResidualResult102149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 102149
end ResidualResult102149

namespace ResidualResult102153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102149.actual selector witness -
    ResidualResult102146.actual selector witness
end ResidualResult102153

namespace ResidualResult102157
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102153.actual selector witness -
    ResidualResult102138.actual selector witness
end ResidualResult102157

namespace ResidualResult102166
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult102019.actual selector witness
end ResidualResult102166

namespace ResidualResult102173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102166.actual selector witness +
    ResidualResult102012.actual selector witness
end ResidualResult102173

namespace ResidualResult102178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102173.actual selector witness +
    ResidualResult101739.actual selector witness
end ResidualResult102178

namespace ResidualResult102183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102178.actual selector witness +
    ResidualResult101305.actual selector witness
end ResidualResult102183

namespace ResidualResult102188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102183.actual selector witness +
    ResidualResult100871.actual selector witness
end ResidualResult102188

namespace ResidualResult102193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102188.actual selector witness +
    ResidualResult100437.actual selector witness
end ResidualResult102193

namespace ResidualResult102198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102193.actual selector witness +
    ResidualResult100003.actual selector witness
end ResidualResult102198

namespace ResidualResult102203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult102198.actual selector witness +
    ResidualResult99569.actual selector witness
end ResidualResult102203

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
