import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard683
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard743

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult104474
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104474
end ResidualResult104474

namespace ResidualResult104478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104474.actual selector witness -
    ResidualResult104471.actual selector witness
end ResidualResult104478

namespace ResidualResult104486
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104478.actual selector witness *
    ResidualResult104455.actual selector witness
end ResidualResult104486

namespace ResidualResult104489
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104489
end ResidualResult104489

namespace ResidualResult104494
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104466.actual selector witness *
    ResidualResult104489.actual selector witness
end ResidualResult104494

namespace ResidualResult104497
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104497
end ResidualResult104497

namespace ResidualResult104501
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104497.actual selector witness -
    ResidualResult104494.actual selector witness
end ResidualResult104501

namespace ResidualResult104505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104501.actual selector witness -
    ResidualResult104486.actual selector witness
end ResidualResult104505

namespace ResidualResult104514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult104367.actual selector witness
end ResidualResult104514

namespace ResidualResult104521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104514.actual selector witness +
    ResidualResult104360.actual selector witness
end ResidualResult104521

namespace ResidualResult104531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult104521.actual selector witness *
    ResidualResult5579.actual selector witness
end ResidualResult104531

namespace ResidualResult104535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104535
end ResidualResult104535

namespace ResidualResult104538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104538
end ResidualResult104538

namespace ResidualResult104548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult96360.actual selector witness *
    ResidualResult104538.actual selector witness
end ResidualResult104548

namespace ResidualResult104551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104551
end ResidualResult104551

namespace ResidualResult104555
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 104555
end ResidualResult104555

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
