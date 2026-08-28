import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult41840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41840
end ResidualResult41840

namespace ResidualResult41845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1866.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult41845

namespace ResidualResult41850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult12484.actual selector witness
end ResidualResult41850

namespace ResidualResult41854
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41850.actual selector witness -
    ResidualResult41845.actual selector witness
end ResidualResult41854

namespace ResidualResult41860
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41854.actual selector witness +
    ResidualResult12476.actual selector witness
end ResidualResult41860

namespace ResidualResult41868
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41860.actual selector witness *
    ResidualResult1869.actual selector witness
end ResidualResult41868

namespace ResidualResult41873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1869.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult41873

namespace ResidualResult41878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult12525.actual selector witness
end ResidualResult41878

namespace ResidualResult41882
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41878.actual selector witness -
    ResidualResult41873.actual selector witness
end ResidualResult41882

namespace ResidualResult41888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41882.actual selector witness +
    ResidualResult12517.actual selector witness
end ResidualResult41888

namespace ResidualResult41898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41888.actual selector witness *
    ResidualResult12514.actual selector witness
end ResidualResult41898

namespace ResidualResult41904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41898.actual selector witness +
    ResidualResult41868.actual selector witness
end ResidualResult41904

namespace ResidualResult41914
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41904.actual selector witness *
    ResidualResult41840.actual selector witness
end ResidualResult41914

namespace ResidualResult41917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41917
end ResidualResult41917

namespace ResidualResult41921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41921
end ResidualResult41921

namespace ResidualResult41999
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41999
end ResidualResult41999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
