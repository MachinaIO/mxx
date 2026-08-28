import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard662

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult93921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93921
end ResidualResult93921

namespace ResidualResult93930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93930
end ResidualResult93930

namespace ResidualResult93932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93932
end ResidualResult93932

namespace ResidualResult93937
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93932.actual selector witness *
    ResidualResult93930.actual selector witness
end ResidualResult93937

namespace ResidualResult93940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93940
end ResidualResult93940

namespace ResidualResult93944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93940.actual selector witness -
    ResidualResult93937.actual selector witness
end ResidualResult93944

namespace ResidualResult93952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93944.actual selector witness *
    ResidualResult93921.actual selector witness
end ResidualResult93952

namespace ResidualResult93955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93955
end ResidualResult93955

namespace ResidualResult93960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93932.actual selector witness *
    ResidualResult93955.actual selector witness
end ResidualResult93960

namespace ResidualResult93963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93963
end ResidualResult93963

namespace ResidualResult93967
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93963.actual selector witness -
    ResidualResult93960.actual selector witness
end ResidualResult93967

namespace ResidualResult93971
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93967.actual selector witness -
    ResidualResult93952.actual selector witness
end ResidualResult93971

namespace ResidualResult93980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult93809.actual selector witness
end ResidualResult93980

namespace ResidualResult93987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93980.actual selector witness +
    ResidualResult93802.actual selector witness
end ResidualResult93987

namespace ResidualResult93997
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93987.actual selector witness *
    ResidualResult5859.actual selector witness
end ResidualResult93997

namespace ResidualResult94002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult94002

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
