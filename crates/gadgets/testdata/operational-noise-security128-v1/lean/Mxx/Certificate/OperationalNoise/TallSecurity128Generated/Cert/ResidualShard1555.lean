import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1457
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1458
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1549
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1551
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1552
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1553
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1554

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult221607
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 221607
end ResidualResult221607

namespace ResidualResult221611
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221607.actual selector witness -
    ResidualResult221604.actual selector witness
end ResidualResult221611

namespace ResidualResult221615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221611.actual selector witness -
    ResidualResult221596.actual selector witness
end ResidualResult221615

namespace ResidualResult221624
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult207620.actual selector witness *
    ResidualResult221453.actual selector witness
end ResidualResult221624

namespace ResidualResult221631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221624.actual selector witness +
    ResidualResult221446.actual selector witness
end ResidualResult221631

namespace ResidualResult221641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221631.actual selector witness *
    ResidualResult15882.actual selector witness
end ResidualResult221641

namespace ResidualResult221646
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult207528.actual selector witness
end ResidualResult221646

namespace ResidualResult221651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult207398.actual selector witness *
    ResidualResult15896.actual selector witness
end ResidualResult221651

namespace ResidualResult221655
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221651.actual selector witness -
    ResidualResult221646.actual selector witness
end ResidualResult221655

namespace ResidualResult221661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221655.actual selector witness +
    ResidualResult31516.actual selector witness
end ResidualResult221661

namespace ResidualResult221668
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221661.actual selector witness -
    ResidualResult221661.actual selector witness
end ResidualResult221668

namespace ResidualResult221673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221668.actual selector witness +
    ResidualResult221641.actual selector witness
end ResidualResult221673

namespace ResidualResult221678
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221673.actual selector witness +
    ResidualResult221429.actual selector witness
end ResidualResult221678

namespace ResidualResult221683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221678.actual selector witness +
    ResidualResult221217.actual selector witness
end ResidualResult221683

namespace ResidualResult221688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221683.actual selector witness +
    ResidualResult221005.actual selector witness
end ResidualResult221688

namespace ResidualResult221693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult221688.actual selector witness +
    ResidualResult220793.actual selector witness
end ResidualResult221693

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
