pub mod clear;
pub mod ctree;
pub mod dataset;
pub mod forest;
pub mod serial;
pub mod tree;

use revolut::radix::{ByteByteLUT, NyblByteLUT};
use revolut::{key, Context, PublicKey};
// REVOLUT
pub use revolut::{radix::ByteLWE, LUT, LWE};

// TFHE
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

// type LWE = LweCiphertext<Vec<u64>>;
type RLWE = GlweCiphertext<Vec<u64>>;

trait Majority {
    fn blind_count_extra(&self, lwes: &[LWE], ctx: &Context) -> Vec<ByteLWE>;
    fn blind_majority_extra(&self, lwes: &[LWE], ctx: &Context) -> LWE;
}

impl Majority for PublicKey {
    /// Count the number of occurences of more than p LWEs
    fn blind_count_extra(&self, lwes: &[LWE], ctx: &Context) -> Vec<ByteLWE> {
        let mut count = NyblByteLUT::from_bytes_trivially(&[0u8; 16], ctx);

        // Make chunks of size p
        let mut chunks = Vec::new();
        let chunk_size = ctx.full_message_modulus() as usize;
        for lwe_chunk in lwes.chunks(chunk_size) {
            chunks.push(lwe_chunk.to_vec());
        }

        for (i, chunk) in chunks.iter().enumerate() {
            let lut = LUT::from_vec_of_lwe(chunk, self, ctx);
            for j in 0..chunk.len() {
                let e = self.lut_extract(&lut, j, ctx);
                count.blind_array_inc(&e, ctx, self);
            }
        }
        count.to_many_blwes(self, ctx)
    }

    /// Blind Majority of more than p LWEs
    fn blind_majority_extra(&self, lwes: &[LWE], ctx: &Context) -> LWE {
        let count = self.blind_count_extra(lwes, ctx);
        let maj = self.blind_argmax_byte_lwe(&count, ctx);
        maj.lo
    }
}

mod tests {
    use std::time::Instant;

    use revolut::key;

    use super::*;

    #[test]
    fn test_blind_majority_extra() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;
        let p = ctx.full_message_modulus() as usize;
        let mut vec = (0..64)
            .map(|_| rand::random::<usize>() % p)
            .collect::<Vec<_>>();

        let expected = {
            let mut counts = std::collections::HashMap::new();
            for &value in &vec {
                *counts.entry(value).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(value, _)| value)
                .unwrap()
        };

        // Encrypt
        let lwes = vec
            .iter()
            .map(|x| private_key.allocate_and_encrypt_lwe(*x as u64, &mut ctx))
            .collect::<Vec<_>>();

        // Blind majority
        let start = Instant::now();
        let maj = public_key.blind_majority_extra(&lwes, &ctx);
        let end = Instant::now();
        println!("time taken: {:?}", end.duration_since(start));

        // Decrypt
        let actual = private_key.decrypt_lwe(&maj, &ctx);
        println!("maj: {}", actual);
        assert_eq!(actual, expected as u64);
    }
}
