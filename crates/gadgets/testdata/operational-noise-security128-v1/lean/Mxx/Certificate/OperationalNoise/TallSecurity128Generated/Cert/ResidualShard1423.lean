import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1358
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1393
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1397
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1401
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1405
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1408
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1412
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1416
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1419
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1422

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult201535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201527.actual selector witness *
    ResidualResult201504.actual selector witness
end ResidualResult201535

namespace ResidualResult201538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 201538
end ResidualResult201538

namespace ResidualResult201543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201515.actual selector witness *
    ResidualResult201538.actual selector witness
end ResidualResult201543

namespace ResidualResult201546
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 201546
end ResidualResult201546

namespace ResidualResult201550
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201546.actual selector witness -
    ResidualResult201543.actual selector witness
end ResidualResult201550

namespace ResidualResult201554
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201550.actual selector witness -
    ResidualResult201535.actual selector witness
end ResidualResult201554

namespace ResidualResult201563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult192995.actual selector witness *
    ResidualResult201392.actual selector witness
end ResidualResult201563

namespace ResidualResult201570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201563.actual selector witness +
    ResidualResult201385.actual selector witness
end ResidualResult201570

namespace ResidualResult201575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201570.actual selector witness +
    ResidualResult201088.actual selector witness
end ResidualResult201575

namespace ResidualResult201580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201575.actual selector witness +
    ResidualResult200606.actual selector witness
end ResidualResult201580

namespace ResidualResult201585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201580.actual selector witness +
    ResidualResult200124.actual selector witness
end ResidualResult201585

namespace ResidualResult201590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201585.actual selector witness +
    ResidualResult199642.actual selector witness
end ResidualResult201590

namespace ResidualResult201595
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201590.actual selector witness +
    ResidualResult199160.actual selector witness
end ResidualResult201595

namespace ResidualResult201600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201595.actual selector witness +
    ResidualResult198678.actual selector witness
end ResidualResult201600

namespace ResidualResult201605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201600.actual selector witness +
    ResidualResult198196.actual selector witness
end ResidualResult201605

namespace ResidualResult201610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult201605.actual selector witness +
    ResidualResult197714.actual selector witness
end ResidualResult201610

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
