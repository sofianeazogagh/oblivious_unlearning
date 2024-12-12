use std::vec;
use csv;

use revolut::{Context, PrivateKey, LUT};
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

type LWE = LweCiphertext<Vec<u64>>;

pub struct EncryptedDataset {
    records: Vec<Vec<LweCiphertext<Vec<u64>>>>,
    labels: Vec<LweCiphertext<Vec<u64>>>,
    f: u64,
    n: u64,
}

impl EncryptedDataset {
    pub fn from_file(filepath: String, private_key: &PrivateKey, ctx: &Context) -> Self {
        let mut records = Vec::new();
        let mut labels = Vec::new();
        let mut f = 0;
        let mut n = 0;

        let mut reader = csv::Reader::from_path(filepath).unwrap();
        for result in reader.records() {
            let record = result.unwrap();
            let mut rec = Vec::new();
            for (i, field) in record.iter().enumerate() {
                if i == record.len() - 1 {
                    let label = field.parse::<u64>().unwrap();
                    let label = private_key.allocate_and_encrypt_lwe(label, &mut ctx);
                    labels.push(label);
                } else {
                    let feature = field.parse::<u64>().unwrap();
                    let feature = private_key.allocate_and_encrypt_lwe(feature, &mut ctx);
                    rec.push(feature);
                }
            }
            records.push(rec);
           
        }

        f = records[0].len() as u64;
        n = records.len() as u64;

        EncryptedDataset {
            records,
            labels,
            f,
            n,
        }
    }

    fn record(self, i: u64) -> Vec<LweCiphertext<Vec<u64>>> {
        let mut rec = Vec::new();
        for j in 0..self.f {
            rec.push(self.records[i as usize][j as usize].clone());
        }

        rec
    }

    fn label(self, i: u64) -> LweCiphertext<Vec<u64>> {
        self.labels[i as usize].clone()
    }
}
