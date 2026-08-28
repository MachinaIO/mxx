import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard167

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult21923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult6973.actual selector witness
end ResidualResult21923

namespace ResidualResult21927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21923.actual selector witness -
    ResidualResult21918.actual selector witness
end ResidualResult21927

namespace ResidualResult21933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21927.actual selector witness +
    ResidualResult6965.actual selector witness
end ResidualResult21933

namespace ResidualResult21941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21933.actual selector witness *
    ResidualResult868.actual selector witness
end ResidualResult21941

namespace ResidualResult21946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult868.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult21946

namespace ResidualResult21951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult7014.actual selector witness
end ResidualResult21951

namespace ResidualResult21955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21951.actual selector witness -
    ResidualResult21946.actual selector witness
end ResidualResult21955

namespace ResidualResult21961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21955.actual selector witness +
    ResidualResult7006.actual selector witness
end ResidualResult21961

namespace ResidualResult21971
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21961.actual selector witness *
    ResidualResult7003.actual selector witness
end ResidualResult21971

namespace ResidualResult21977
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21971.actual selector witness +
    ResidualResult21941.actual selector witness
end ResidualResult21977

namespace ResidualResult21987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21977.actual selector witness *
    ResidualResult21913.actual selector witness
end ResidualResult21987

namespace ResidualResult21990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 21990
end ResidualResult21990

namespace ResidualResult21994
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 21994
end ResidualResult21994

namespace ResidualResult22072
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 22072
end ResidualResult22072

namespace ResidualResult22075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 22075
end ResidualResult22075

namespace ResidualResult22080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult22075.actual selector witness *
    ResidualResult22072.actual selector witness
end ResidualResult22080

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
