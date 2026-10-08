//! GPU round trip of the scheme: every stage on the GPU runtime.
use crate::{
    BranchingProgram, PeClass, decrypt, encrypt, keygen, setup,
    test_support::{equals, parameters},
};
use mxx_backends::{
    GpuRuntime, MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
};

/// One setup, one ciphertext of a random message, and keys for an
/// authorized and an unauthorized equality predicate, all on the GPU, with
/// parameters whose simulated bound guarantees correctness (by default the
/// noise-free test parameters): the authorized key decrypts the message,
/// and the unauthorized decryption returns a rounded phase, which this test
/// does not constrain.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_authorized_key_decrypts_on_the_gpu() {
    let class = PeClass::new(2, 2, 2, 2).unwrap();
    let parameters = parameters(class);
    let level = parameters.level();
    let mut execution = GpuRuntime::new(gpu_backend([GpuDCRTPolyParams::new(
        level.ring_dimension(),
        level.moduli().to_vec(),
        level.base_bits(),
        None,
    )]))
    .expect("GPU runtime");
    let bound = crate::noise::simulate(&parameters, &BranchingProgram::longest(class)).unwrap();
    assert!(bound.decrypts(), "the parameters must guarantee correctness");
    let mut store = MemoryArtifactStore::default();
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    let attribute = [0; 2].map(|_| rand::random::<u32>() as usize % 2);
    let message = (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let ciphertext =
        encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
    let complement = attribute.map(|bit| 1 - bit);
    let predicates: [BranchingProgram; 2] = [equals(class, &attribute), equals(class, &complement)];
    for (predicate, value) in predicates.iter().zip([0, 1]) {
        assert_eq!(predicate.evaluate(&attribute).unwrap(), value);
        let key = keygen(&public_key, &master, predicate, &mut execution, &mut store).unwrap();
        let decrypted =
            decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
        if value == 0 {
            assert_eq!(decrypted, message);
        }
    }
}
