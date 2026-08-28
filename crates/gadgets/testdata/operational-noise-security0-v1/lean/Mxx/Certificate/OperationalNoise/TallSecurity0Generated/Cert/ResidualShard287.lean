import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard286

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult38958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult9478.actual selector witness
end ResidualResult38958

namespace ResidualResult38962
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38958.actual selector witness -
    ResidualResult38953.actual selector witness
end ResidualResult38962

namespace ResidualResult38968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38962.actual selector witness +
    ResidualResult9470.actual selector witness
end ResidualResult38968

namespace ResidualResult38976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38968.actual selector witness *
    ResidualResult1731.actual selector witness
end ResidualResult38976

namespace ResidualResult38981
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1731.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult38981

namespace ResidualResult38986
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult9519.actual selector witness
end ResidualResult38986

namespace ResidualResult38990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38986.actual selector witness -
    ResidualResult38981.actual selector witness
end ResidualResult38990

namespace ResidualResult38996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38990.actual selector witness +
    ResidualResult9511.actual selector witness
end ResidualResult38996

namespace ResidualResult39006
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38996.actual selector witness *
    ResidualResult9508.actual selector witness
end ResidualResult39006

namespace ResidualResult39012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39006.actual selector witness +
    ResidualResult38976.actual selector witness
end ResidualResult39012

namespace ResidualResult39022
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39012.actual selector witness *
    ResidualResult38948.actual selector witness
end ResidualResult39022

namespace ResidualResult39025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39025
end ResidualResult39025

namespace ResidualResult39029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39029
end ResidualResult39029

namespace ResidualResult39107
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39107
end ResidualResult39107

namespace ResidualResult39110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39110
end ResidualResult39110

namespace ResidualResult39115
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39110.actual selector witness *
    ResidualResult39107.actual selector witness
end ResidualResult39115

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
