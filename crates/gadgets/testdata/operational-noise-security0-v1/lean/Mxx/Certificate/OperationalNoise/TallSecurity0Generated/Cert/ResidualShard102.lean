import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard101

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult12009
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12009
end ResidualResult12009

namespace ResidualResult12013
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12013
end ResidualResult12013

namespace ResidualResult12016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12016
end ResidualResult12016

namespace ResidualResult12021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult307.actual selector witness *
    ResidualResult6449.actual selector witness
end ResidualResult12021

namespace ResidualResult12024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12024
end ResidualResult12024

namespace ResidualResult12029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6314.actual selector witness *
    ResidualResult12024.actual selector witness
end ResidualResult12029

namespace ResidualResult12033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12029.actual selector witness -
    ResidualResult12021.actual selector witness
end ResidualResult12033

namespace ResidualResult12039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12033.actual selector witness +
    ResidualResult12016.actual selector witness
end ResidualResult12039

namespace ResidualResult12049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12039.actual selector witness *
    ResidualResult12013.actual selector witness
end ResidualResult12049

namespace ResidualResult12055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12049.actual selector witness +
    ResidualResult12006.actual selector witness
end ResidualResult12055

namespace ResidualResult12065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12055.actual selector witness *
    ResidualResult11972.actual selector witness
end ResidualResult12065

namespace ResidualResult12068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12068
end ResidualResult12068

namespace ResidualResult12072
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12072
end ResidualResult12072

namespace ResidualResult12150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12150
end ResidualResult12150

namespace ResidualResult12153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 12153
end ResidualResult12153

namespace ResidualResult12158
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult12153.actual selector witness *
    ResidualResult12150.actual selector witness
end ResidualResult12158

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
