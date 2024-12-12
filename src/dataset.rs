use std::vec;
use csv;

//  import Query from probonite.rs
use crate::probonite::Query;


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


}
