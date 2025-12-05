pub mod digits_to_int;
pub mod encoding;
pub mod public_key;
pub mod sampler;

#[cfg(test)]
mod tests {
    use crate::{
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{hash::DCRTPolyHashSampler, uniform::DCRTPolyUniformSampler},
        utils::{create_random_poly, create_ternary_random_poly},
    };
    use keccak_asm::Keccak256;

    #[test]
    fn test_bgg_pub_key_addition() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let log_base_q = params.modulus_digits();
        let columns = d * log_base_q;

        let a = sampled_pub_keys[1].clone();
        let b = sampled_pub_keys[2].clone();
        let addition = a.clone() + b.clone();
        assert_eq!(addition.matrix.row_size(), d);
        assert_eq!(addition.matrix.col_size(), columns);
        assert_eq!(addition.matrix, a.matrix.clone() + b.matrix.clone());
    }

    #[test]
    fn test_bgg_pub_key_multiplication() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let log_base_q = params.modulus_digits();
        let columns = d * log_base_q;

        let a = sampled_pub_keys[1].clone();
        let b = sampled_pub_keys[2].clone();
        let multiplication = a.clone() * b.clone();
        assert_eq!(multiplication.matrix.row_size(), d);
        assert_eq!(multiplication.matrix.col_size(), columns);
        assert_eq!(multiplication.matrix, (a.matrix.clone() * b.matrix.decompose().clone()))
    }

    #[test]
    fn test_bgg_encoding_sampling() {
        let input_size = 10_usize;
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = input_size.div_ceil(params.ring_dimension().try_into().unwrap());
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![DCRTPoly::const_one(&params); packed_input_size];
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);
        let g = DCRTPolyMatrix::gadget_matrix(&params, d);
        assert_eq!(bgg_encodings.len(), packed_input_size + 1);
        assert_eq!(
            bgg_encodings[0].vector,
            bgg_sampler.secret_vec.clone() * bgg_encodings[0].pubkey.matrix.clone() -
                bgg_sampler.secret_vec.clone() *
                    (g.clone() * bgg_encodings[0].plaintext.clone().unwrap())
        );
        assert_eq!(
            bgg_encodings[1].vector,
            bgg_sampler.secret_vec.clone() * bgg_encodings[1].pubkey.matrix.clone() -
                bgg_sampler.secret_vec.clone() *
                    (g * bgg_encodings[1].plaintext.clone().unwrap())
        )
    }

    #[test]
    fn test_bgg_encoding_addition() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![create_random_poly(&params); packed_input_size];
        // TODO: set the standard deviation to a non-zero value
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);

        for pair in bgg_encodings[1..].chunks(2) {
            if let [a, b] = pair {
                let addition = a.clone() + b.clone();
                assert_eq!(addition.pubkey, a.pubkey.clone() + b.pubkey.clone());
                assert_eq!(
                    addition.clone().plaintext.unwrap(),
                    a.plaintext.clone().unwrap() + b.plaintext.clone().unwrap()
                );
                let g = DCRTPolyMatrix::gadget_matrix(&params, d);
                assert_eq!(addition.vector, a.clone().vector + b.clone().vector);
                assert_eq!(
                    addition.vector,
                    bgg_sampler.secret_vec.clone() *
                        (addition.pubkey.matrix - (g * addition.plaintext.unwrap()))
                )
            }
        }
    }

    #[test]
    fn test_bgg_encoding_multiplication() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let params = DCRTPolyParams::default();
        let packed_input_size = 2;
        let d = 3;
        let bgg_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let reveal_plaintexts = vec![true; packed_input_size];
        let sampled_pub_keys = bgg_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let secrets = vec![create_ternary_random_poly(&params); d];
        let plaintexts = vec![create_random_poly(&params); packed_input_size];
        let bgg_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let bgg_encodings = bgg_sampler.sample(&params, &sampled_pub_keys, &plaintexts);

        for pair in bgg_encodings[1..].chunks(2) {
            if let [a, b] = pair {
                let multiplication = a.clone() * b.clone();
                assert_eq!(multiplication.pubkey, (a.clone().pubkey * b.clone().pubkey));
                assert_eq!(
                    multiplication.clone().plaintext.unwrap(),
                    a.clone().plaintext.unwrap() * b.clone().plaintext.unwrap()
                );
                let g = DCRTPolyMatrix::gadget_matrix(&params, d);
                assert_eq!(
                    multiplication.vector,
                    (bgg_sampler.secret_vec.clone() *
                        (multiplication.pubkey.matrix - (g * multiplication.plaintext.unwrap())))
                )
            }
        }
    }
}
