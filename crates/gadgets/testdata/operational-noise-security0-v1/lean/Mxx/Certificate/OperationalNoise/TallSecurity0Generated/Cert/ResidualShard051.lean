import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard043
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard050

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult5938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5934.actual selector witness +
    ResidualResult5587.actual selector witness
end ResidualResult5938

namespace ResidualResult5942
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5938.actual selector witness +
    ResidualResult5567.actual selector witness
end ResidualResult5942

namespace ResidualResult5946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5942.actual selector witness +
    ResidualResult5547.actual selector witness
end ResidualResult5946

namespace ResidualResult5950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5946.actual selector witness +
    ResidualResult5527.actual selector witness
end ResidualResult5950

namespace ResidualResult5954
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5950.actual selector witness -
    ResidualResult5507.actual selector witness
end ResidualResult5954

namespace ResidualResult5957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5957
end ResidualResult5957

namespace ResidualResult5961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5961
end ResidualResult5961

namespace ResidualResult5964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5964
end ResidualResult5964

namespace ResidualResult5969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5964.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult5969

namespace ResidualResult5974
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5969.actual selector witness *
    ResidualResult5487.actual selector witness
end ResidualResult5974

namespace ResidualResult5979
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5974.actual selector witness *
    ResidualResult5476.actual selector witness
end ResidualResult5979

namespace ResidualResult5984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2.actual selector witness *
    ResidualResult829.actual selector witness
end ResidualResult5984

namespace ResidualResult5987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5987
end ResidualResult5987

namespace ResidualResult5991
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5991
end ResidualResult5991

namespace ResidualResult5994
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5994
end ResidualResult5994

namespace ResidualResult5997
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5997
end ResidualResult5997

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
