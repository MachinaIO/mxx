import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard588
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard589

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult83011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83011
end ResidualResult83011

namespace ResidualResult83017
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83017
end ResidualResult83017

namespace ResidualResult83021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83021
end ResidualResult83021

namespace ResidualResult83024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83024
end ResidualResult83024

namespace ResidualResult83029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83024.actual selector witness *
    ResidualResult83021.actual selector witness
end ResidualResult83029

namespace ResidualResult83033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83029.actual selector witness -
    ResidualResult83008.actual selector witness
end ResidualResult83033

namespace ResidualResult83041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83033.actual selector witness *
    ResidualResult82992.actual selector witness
end ResidualResult83041

namespace ResidualResult83044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83044
end ResidualResult83044

namespace ResidualResult83049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83003.actual selector witness *
    ResidualResult83044.actual selector witness
end ResidualResult83049

namespace ResidualResult83052
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83052
end ResidualResult83052

namespace ResidualResult83056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83052.actual selector witness -
    ResidualResult83049.actual selector witness
end ResidualResult83056

namespace ResidualResult83060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83056.actual selector witness -
    ResidualResult83041.actual selector witness
end ResidualResult83060

namespace ResidualResult83069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult82892.actual selector witness
end ResidualResult83069

namespace ResidualResult83076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83069.actual selector witness +
    ResidualResult82885.actual selector witness
end ResidualResult83076

namespace ResidualResult83086
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult83076.actual selector witness *
    ResidualResult82801.actual selector witness
end ResidualResult83086

namespace ResidualResult83089
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 83089
end ResidualResult83089

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
