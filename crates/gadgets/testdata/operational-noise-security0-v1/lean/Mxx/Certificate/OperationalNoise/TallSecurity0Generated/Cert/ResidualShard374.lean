import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard373

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult51837
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51837
end ResidualResult51837

namespace ResidualResult51842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51837.actual selector witness *
    ResidualResult51835.actual selector witness
end ResidualResult51842

namespace ResidualResult51847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51847
end ResidualResult51847

namespace ResidualResult51853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51853
end ResidualResult51853

namespace ResidualResult51857
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51857
end ResidualResult51857

namespace ResidualResult51860
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51860
end ResidualResult51860

namespace ResidualResult51865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51860.actual selector witness *
    ResidualResult51857.actual selector witness
end ResidualResult51865

namespace ResidualResult51869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51865.actual selector witness -
    ResidualResult51842.actual selector witness
end ResidualResult51869

namespace ResidualResult51877
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51869.actual selector witness *
    ResidualResult51826.actual selector witness
end ResidualResult51877

namespace ResidualResult51880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51880
end ResidualResult51880

namespace ResidualResult51885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51837.actual selector witness *
    ResidualResult51880.actual selector witness
end ResidualResult51885

namespace ResidualResult51888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51888
end ResidualResult51888

namespace ResidualResult51892
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51888.actual selector witness -
    ResidualResult51885.actual selector witness
end ResidualResult51892

namespace ResidualResult51896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51892.actual selector witness -
    ResidualResult51877.actual selector witness
end ResidualResult51896

namespace ResidualResult51905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult51726.actual selector witness
end ResidualResult51905

namespace ResidualResult51912
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51905.actual selector witness +
    ResidualResult51719.actual selector witness
end ResidualResult51912

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
