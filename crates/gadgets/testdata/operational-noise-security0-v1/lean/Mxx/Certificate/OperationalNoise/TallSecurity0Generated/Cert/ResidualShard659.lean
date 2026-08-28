import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard624
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard658

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult93296
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93296
end ResidualResult93296

namespace ResidualResult93301
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93296.actual selector witness *
    ResidualResult93294.actual selector witness
end ResidualResult93301

namespace ResidualResult93304
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93304
end ResidualResult93304

namespace ResidualResult93308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93304.actual selector witness -
    ResidualResult93301.actual selector witness
end ResidualResult93308

namespace ResidualResult93316
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93308.actual selector witness *
    ResidualResult93285.actual selector witness
end ResidualResult93316

namespace ResidualResult93319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93319
end ResidualResult93319

namespace ResidualResult93324
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93296.actual selector witness *
    ResidualResult93319.actual selector witness
end ResidualResult93324

namespace ResidualResult93327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93327
end ResidualResult93327

namespace ResidualResult93331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93327.actual selector witness -
    ResidualResult93324.actual selector witness
end ResidualResult93331

namespace ResidualResult93335
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93331.actual selector witness -
    ResidualResult93316.actual selector witness
end ResidualResult93335

namespace ResidualResult93344
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult93173.actual selector witness
end ResidualResult93344

namespace ResidualResult93351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93344.actual selector witness +
    ResidualResult93166.actual selector witness
end ResidualResult93351

namespace ResidualResult93361
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult93351.actual selector witness *
    ResidualResult5799.actual selector witness
end ResidualResult93361

namespace ResidualResult93365
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93365
end ResidualResult93365

namespace ResidualResult93368
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 93368
end ResidualResult93368

namespace ResidualResult93378
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult87396.actual selector witness *
    ResidualResult93368.actual selector witness
end ResidualResult93378

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
