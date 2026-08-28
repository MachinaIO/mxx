import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard404
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard450

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult63022
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63022
end ResidualResult63022

namespace ResidualResult63027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63022.actual selector witness *
    ResidualResult63020.actual selector witness
end ResidualResult63027

namespace ResidualResult63030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63030
end ResidualResult63030

namespace ResidualResult63034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63030.actual selector witness -
    ResidualResult63027.actual selector witness
end ResidualResult63034

namespace ResidualResult63042
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63034.actual selector witness *
    ResidualResult63011.actual selector witness
end ResidualResult63042

namespace ResidualResult63045
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63045
end ResidualResult63045

namespace ResidualResult63050
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63022.actual selector witness *
    ResidualResult63045.actual selector witness
end ResidualResult63050

namespace ResidualResult63053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63053
end ResidualResult63053

namespace ResidualResult63057
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63053.actual selector witness -
    ResidualResult63050.actual selector witness
end ResidualResult63057

namespace ResidualResult63061
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63057.actual selector witness -
    ResidualResult63042.actual selector witness
end ResidualResult63061

namespace ResidualResult63070
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult62899.actual selector witness
end ResidualResult63070

namespace ResidualResult63077
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63070.actual selector witness +
    ResidualResult62892.actual selector witness
end ResidualResult63077

namespace ResidualResult63087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult63077.actual selector witness *
    ResidualResult5699.actual selector witness
end ResidualResult63087

namespace ResidualResult63091
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63091
end ResidualResult63091

namespace ResidualResult63094
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 63094
end ResidualResult63094

namespace ResidualResult63104
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55768.actual selector witness *
    ResidualResult63094.actual selector witness
end ResidualResult63104

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
