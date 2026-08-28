import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard290
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard291

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult39632
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39632
end ResidualResult39632

namespace ResidualResult39638
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39638
end ResidualResult39638

namespace ResidualResult39642
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39642
end ResidualResult39642

namespace ResidualResult39645
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39645
end ResidualResult39645

namespace ResidualResult39650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39645.actual selector witness *
    ResidualResult39642.actual selector witness
end ResidualResult39650

namespace ResidualResult39654
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39650.actual selector witness -
    ResidualResult39627.actual selector witness
end ResidualResult39654

namespace ResidualResult39662
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39654.actual selector witness *
    ResidualResult39611.actual selector witness
end ResidualResult39662

namespace ResidualResult39665
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39665
end ResidualResult39665

namespace ResidualResult39670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39622.actual selector witness *
    ResidualResult39665.actual selector witness
end ResidualResult39670

namespace ResidualResult39673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39673
end ResidualResult39673

namespace ResidualResult39677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39673.actual selector witness -
    ResidualResult39670.actual selector witness
end ResidualResult39677

namespace ResidualResult39681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39677.actual selector witness -
    ResidualResult39662.actual selector witness
end ResidualResult39681

namespace ResidualResult39690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult39511.actual selector witness
end ResidualResult39690

namespace ResidualResult39697
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39690.actual selector witness +
    ResidualResult39504.actual selector witness
end ResidualResult39697

namespace ResidualResult39707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult39697.actual selector witness *
    ResidualResult39420.actual selector witness
end ResidualResult39707

namespace ResidualResult39710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 39710
end ResidualResult39710

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
