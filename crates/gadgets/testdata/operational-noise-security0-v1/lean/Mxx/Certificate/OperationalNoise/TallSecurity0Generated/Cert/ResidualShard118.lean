import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult13992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6314.actual selector witness *
    ResidualResult13987.actual selector witness
end ResidualResult13992

namespace ResidualResult13996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult13992.actual selector witness -
    ResidualResult13984.actual selector witness
end ResidualResult13996

namespace ResidualResult14002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult13996.actual selector witness +
    ResidualResult13979.actual selector witness
end ResidualResult14002

namespace ResidualResult14010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14002.actual selector witness *
    ResidualResult399.actual selector witness
end ResidualResult14010

namespace ResidualResult14013
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14013
end ResidualResult14013

namespace ResidualResult14017
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14017
end ResidualResult14017

namespace ResidualResult14020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14020
end ResidualResult14020

namespace ResidualResult14025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult399.actual selector witness *
    ResidualResult6449.actual selector witness
end ResidualResult14025

namespace ResidualResult14028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14028
end ResidualResult14028

namespace ResidualResult14033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6314.actual selector witness *
    ResidualResult14028.actual selector witness
end ResidualResult14033

namespace ResidualResult14037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14033.actual selector witness -
    ResidualResult14025.actual selector witness
end ResidualResult14037

namespace ResidualResult14043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14037.actual selector witness +
    ResidualResult14020.actual selector witness
end ResidualResult14043

namespace ResidualResult14053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14043.actual selector witness *
    ResidualResult14017.actual selector witness
end ResidualResult14053

namespace ResidualResult14059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14053.actual selector witness +
    ResidualResult14010.actual selector witness
end ResidualResult14059

namespace ResidualResult14069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14059.actual selector witness *
    ResidualResult13976.actual selector witness
end ResidualResult14069

namespace ResidualResult14072
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14072
end ResidualResult14072

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
