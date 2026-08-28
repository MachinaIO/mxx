import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard061

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult94805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 94805
end ResidualResult94805

namespace ResidualResult94812
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 94812
end ResidualResult94812

namespace ResidualResult94815
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 94815
end ResidualResult94815

namespace ResidualResult94820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4589.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult94820

namespace ResidualResult94825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult6973.actual selector witness
end ResidualResult94825

namespace ResidualResult94829
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94825.actual selector witness -
    ResidualResult94820.actual selector witness
end ResidualResult94829

namespace ResidualResult94835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94829.actual selector witness +
    ResidualResult6965.actual selector witness
end ResidualResult94835

namespace ResidualResult94843
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94835.actual selector witness *
    ResidualResult4592.actual selector witness
end ResidualResult94843

namespace ResidualResult94848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4592.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult94848

namespace ResidualResult94853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult7014.actual selector witness
end ResidualResult94853

namespace ResidualResult94857
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94853.actual selector witness -
    ResidualResult94848.actual selector witness
end ResidualResult94857

namespace ResidualResult94863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94857.actual selector witness +
    ResidualResult7006.actual selector witness
end ResidualResult94863

namespace ResidualResult94873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94863.actual selector witness *
    ResidualResult7003.actual selector witness
end ResidualResult94873

namespace ResidualResult94879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94873.actual selector witness +
    ResidualResult94843.actual selector witness
end ResidualResult94879

namespace ResidualResult94889
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94879.actual selector witness *
    ResidualResult94815.actual selector witness
end ResidualResult94889

namespace ResidualResult94892
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 94892
end ResidualResult94892

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
