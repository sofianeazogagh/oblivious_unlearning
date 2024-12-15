use std::vec;
use csv;

//  import Query from probonite.rs
use crate::probonite::Query;
use rand::seq::SliceRandom;


use revolut::{Context, PrivateKey, LUT};
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};


type LWE = LweCiphertext<Vec<u64>>;

pub struct EncryptedDataset {
    pub records: Vec<Query>,
    pub f: u64,
    pub n: u64,
}

impl EncryptedDataset {
    pub fn from_file(filepath: String, private_key: &PrivateKey, ctx: &mut Context) -> Self {
        let mut rdr = csv::Reader::from_path(filepath).unwrap();
        let mut records = Vec::new();
        let mut f = 0;
        let mut n = 0;

        for result in rdr.records() {
            let record = result.unwrap();
            let mut label: u64 = 0;
            let mut record_vec = Vec::new();
            for (i, field) in record.iter().enumerate() {
                if i == record.len() - 1 {
                    label = field.parse::<u64>().unwrap();
                } else {
                    record_vec.push(field.parse::<u64>().unwrap());
                }
            }
            records.push(Query::make_query(&record_vec, &label, private_key, ctx));
            n += 1;
            f = record_vec.len() as u64;
        }

        if f > ctx.full_message_modulus() as u64 {
            panic!("Number of features exceeds the modulus");
        }

    

        Self {
            records,
            f,
            n,
        }
    }

    pub fn split(&self, train: f64) -> (EncryptedDataset, EncryptedDataset) {
        let mut rng = rand::thread_rng();
        let n = self.records.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let train_size = (n as f64 * train).round() as usize;
        let train_indices = &indices[0..train_size];
        let test_indices = &indices[train_size..];

        let train_records: Vec<Query> = train_indices.iter().map(|&i| self.records[i].clone()).collect();
        let test_records: Vec<Query> = test_indices.iter().map(|&i| self.records[i].clone()).collect();

        (
            EncryptedDataset {
                records: train_records,
                f: self.f,
                n: train_size as u64,
            },
            EncryptedDataset {
                records: test_records,
                f: self.f,
                n: (n - train_size) as u64,
            },
        )

    }

}
