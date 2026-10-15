import RuntimeHash

namespace MxxRuntime.Sha256Columns

def exampleSeed : Blob := (List.range 32).map UInt8.ofNat

#guard digest exampleSeed 0 == #[
  0xa9d6e500, 0x293a88bd, 0x38cbe213, 0xd07ab71f,
  0x8cb22585, 0x52072a01, 0xbdf1c40b, 0xe527f4d0]
#guard digest exampleSeed 1 == #[
  0x04ef472d, 0xd8b73b3f, 0x173309f3, 0xa009ee16,
  0x99a53975, 0x53244fc0, 0x6590c665, 0x345eeb45]
#guard digest exampleSeed 257 == #[
  0x05f34ca6, 0x0ca492c7, 0x62b6c097, 0xa6d4875e,
  0x08250087, 0x96966f8f, 0x953de5cb, 0x24158323]
#guard digest exampleSeed 4294967297 == #[
  0xf59b792a, 0x4be6b146, 0xe4b8e782, 0x4793afaa,
  0xc777072b, 0x18aa41d5, 0x4f7402e5, 0x87c69521]

#guard column exampleSeed 19 23 5 0 == (668202157038767136, 12)
#guard column exampleSeed 19 23 5 22 == (628723092529743904, 12)

end MxxRuntime.Sha256Columns
