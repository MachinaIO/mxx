import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard438
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard439
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard442
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard443
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard463

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult64885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64880.actual selector witness +
    ResidualResult61815.actual selector witness
end ResidualResult64885

namespace ResidualResult64890
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64885.actual selector witness +
    ResidualResult61603.actual selector witness
end ResidualResult64890

namespace ResidualResult64895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64890.actual selector witness +
    ResidualResult61391.actual selector witness
end ResidualResult64895

namespace ResidualResult64900
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64895.actual selector witness +
    ResidualResult61179.actual selector witness
end ResidualResult64900

namespace ResidualResult64905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64900.actual selector witness -
    ResidualResult60967.actual selector witness
end ResidualResult64905

namespace ResidualResult64907
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 64907
end ResidualResult64907

namespace ResidualResult64912
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult6084.actual selector witness
end ResidualResult64912

namespace ResidualResult64916
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64912.actual selector witness -
    ResidualResult50670.actual selector witness
end ResidualResult64916

namespace ResidualResult64922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64916.actual selector witness +
    ResidualResult64907.actual selector witness
end ResidualResult64922

namespace ResidualResult64950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64922.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult64950

namespace ResidualResult64974
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64950.actual selector witness +
    ResidualResult64905.actual selector witness
end ResidualResult64974

namespace ResidualResult65038
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult64974.actual selector witness *
    ResidualResult6081.actual selector witness
end ResidualResult65038

namespace ResidualResult65062
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65038.actual selector witness +
    ResidualResult50635.actual selector witness
end ResidualResult65062

namespace ResidualResult65126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65062.actual selector witness *
    ResidualResult6071.actual selector witness
end ResidualResult65126

namespace ResidualResult65128
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65128
end ResidualResult65128

namespace ResidualResult65149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 65149
end ResidualResult65149

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
