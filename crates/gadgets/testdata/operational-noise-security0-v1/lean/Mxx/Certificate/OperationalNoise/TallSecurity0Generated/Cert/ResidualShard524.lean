import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard523

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult72948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72943.actual selector witness *
    ResidualResult72941.actual selector witness
end ResidualResult72948

namespace ResidualResult72951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72951
end ResidualResult72951

namespace ResidualResult72955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72951.actual selector witness -
    ResidualResult72948.actual selector witness
end ResidualResult72955

namespace ResidualResult72963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72955.actual selector witness *
    ResidualResult72932.actual selector witness
end ResidualResult72963

namespace ResidualResult72966
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72966
end ResidualResult72966

namespace ResidualResult72971
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72943.actual selector witness *
    ResidualResult72966.actual selector witness
end ResidualResult72971

namespace ResidualResult72974
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 72974
end ResidualResult72974

namespace ResidualResult72978
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72974.actual selector witness -
    ResidualResult72971.actual selector witness
end ResidualResult72978

namespace ResidualResult72982
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72978.actual selector witness -
    ResidualResult72963.actual selector witness
end ResidualResult72982

namespace ResidualResult72991
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult72820.actual selector witness
end ResidualResult72991

namespace ResidualResult72998
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult72991.actual selector witness +
    ResidualResult72813.actual selector witness
end ResidualResult72998

namespace ResidualResult73005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73005
end ResidualResult73005

namespace ResidualResult73008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73008
end ResidualResult73008

namespace ResidualResult73015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73015
end ResidualResult73015

namespace ResidualResult73018
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 73018
end ResidualResult73018

namespace ResidualResult73023
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3454.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult73023

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
