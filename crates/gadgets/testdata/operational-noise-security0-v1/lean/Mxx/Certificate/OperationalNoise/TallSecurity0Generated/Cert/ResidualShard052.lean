import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard051

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult6001
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6001
end ResidualResult6001

namespace ResidualResult6004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6004
end ResidualResult6004

namespace ResidualResult6009
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6004.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult6009

namespace ResidualResult6014
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6009.actual selector witness *
    ResidualResult6001.actual selector witness
end ResidualResult6014

namespace ResidualResult6019
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6014.actual selector witness *
    ResidualResult5991.actual selector witness
end ResidualResult6019

namespace ResidualResult6024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2.actual selector witness *
    ResidualResult1577.actual selector witness
end ResidualResult6024

namespace ResidualResult6027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6027
end ResidualResult6027

namespace ResidualResult6031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6031
end ResidualResult6031

namespace ResidualResult6034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6034
end ResidualResult6034

namespace ResidualResult6037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6037
end ResidualResult6037

namespace ResidualResult6041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6041
end ResidualResult6041

namespace ResidualResult6044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 6044
end ResidualResult6044

namespace ResidualResult6049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6044.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult6049

namespace ResidualResult6054
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6049.actual selector witness *
    ResidualResult6041.actual selector witness
end ResidualResult6054

namespace ResidualResult6059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6054.actual selector witness *
    ResidualResult6031.actual selector witness
end ResidualResult6059

namespace ResidualResult6064
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2.actual selector witness *
    ResidualResult2325.actual selector witness
end ResidualResult6064

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
