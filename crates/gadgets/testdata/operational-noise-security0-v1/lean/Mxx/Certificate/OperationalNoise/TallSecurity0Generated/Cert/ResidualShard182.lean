import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard181

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult23789
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23789
end ResidualResult23789

namespace ResidualResult23794
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23766.actual selector witness *
    ResidualResult23789.actual selector witness
end ResidualResult23794

namespace ResidualResult23797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23797
end ResidualResult23797

namespace ResidualResult23801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23797.actual selector witness -
    ResidualResult23794.actual selector witness
end ResidualResult23801

namespace ResidualResult23805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23801.actual selector witness -
    ResidualResult23786.actual selector witness
end ResidualResult23805

namespace ResidualResult23814
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult23643.actual selector witness
end ResidualResult23814

namespace ResidualResult23821
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23814.actual selector witness +
    ResidualResult23636.actual selector witness
end ResidualResult23821

namespace ResidualResult23828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23828
end ResidualResult23828

namespace ResidualResult23831
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23831
end ResidualResult23831

namespace ResidualResult23838
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23838
end ResidualResult23838

namespace ResidualResult23841
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 23841
end ResidualResult23841

namespace ResidualResult23846
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult957.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult23846

namespace ResidualResult23851
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult8977.actual selector witness
end ResidualResult23851

namespace ResidualResult23855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23851.actual selector witness -
    ResidualResult23846.actual selector witness
end ResidualResult23855

namespace ResidualResult23861
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23855.actual selector witness +
    ResidualResult8969.actual selector witness
end ResidualResult23861

namespace ResidualResult23869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult23861.actual selector witness *
    ResidualResult960.actual selector witness
end ResidualResult23869

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
