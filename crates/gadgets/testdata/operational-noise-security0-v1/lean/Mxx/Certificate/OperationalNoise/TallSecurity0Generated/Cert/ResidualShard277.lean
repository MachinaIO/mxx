import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult37683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37683
end ResidualResult37683

namespace ResidualResult37692
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37692
end ResidualResult37692

namespace ResidualResult37694
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37694
end ResidualResult37694

namespace ResidualResult37699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37694.actual selector witness *
    ResidualResult37692.actual selector witness
end ResidualResult37699

namespace ResidualResult37704
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37704
end ResidualResult37704

namespace ResidualResult37710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37710
end ResidualResult37710

namespace ResidualResult37714
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37714
end ResidualResult37714

namespace ResidualResult37717
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37717
end ResidualResult37717

namespace ResidualResult37722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37717.actual selector witness *
    ResidualResult37714.actual selector witness
end ResidualResult37722

namespace ResidualResult37726
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37722.actual selector witness -
    ResidualResult37699.actual selector witness
end ResidualResult37726

namespace ResidualResult37734
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37726.actual selector witness *
    ResidualResult37683.actual selector witness
end ResidualResult37734

namespace ResidualResult37737
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37737
end ResidualResult37737

namespace ResidualResult37742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37694.actual selector witness *
    ResidualResult37737.actual selector witness
end ResidualResult37742

namespace ResidualResult37745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 37745
end ResidualResult37745

namespace ResidualResult37749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37745.actual selector witness -
    ResidualResult37742.actual selector witness
end ResidualResult37749

namespace ResidualResult37753
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult37749.actual selector witness -
    ResidualResult37734.actual selector witness
end ResidualResult37753

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
