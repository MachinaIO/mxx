import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard695

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult97858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4750.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult97858

namespace ResidualResult97863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult10480.actual selector witness
end ResidualResult97863

namespace ResidualResult97867
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97863.actual selector witness -
    ResidualResult97858.actual selector witness
end ResidualResult97867

namespace ResidualResult97873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97867.actual selector witness +
    ResidualResult10472.actual selector witness
end ResidualResult97873

namespace ResidualResult97881
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97873.actual selector witness *
    ResidualResult4753.actual selector witness
end ResidualResult97881

namespace ResidualResult97886
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4753.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult97886

namespace ResidualResult97891
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult10521.actual selector witness
end ResidualResult97891

namespace ResidualResult97895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97891.actual selector witness -
    ResidualResult97886.actual selector witness
end ResidualResult97895

namespace ResidualResult97901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97895.actual selector witness +
    ResidualResult10513.actual selector witness
end ResidualResult97901

namespace ResidualResult97911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97901.actual selector witness *
    ResidualResult10510.actual selector witness
end ResidualResult97911

namespace ResidualResult97917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97911.actual selector witness +
    ResidualResult97881.actual selector witness
end ResidualResult97917

namespace ResidualResult97927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult97917.actual selector witness *
    ResidualResult97853.actual selector witness
end ResidualResult97927

namespace ResidualResult97930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97930
end ResidualResult97930

namespace ResidualResult97934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97934
end ResidualResult97934

namespace ResidualResult97988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97988
end ResidualResult97988

namespace ResidualResult97991
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 97991
end ResidualResult97991

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
