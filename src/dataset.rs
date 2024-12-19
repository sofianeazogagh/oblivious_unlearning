use csv;
use std::vec;

//  import Query from probonite.rs
use crate::probonite::Query;
use rand::seq::SliceRandom;

use crate::clear_model::ClearDataset;
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

        Self { records, f, n }
    }

    pub fn split(&self, train: f64) -> (EncryptedDataset, EncryptedDataset) {
        let mut rng = rand::thread_rng();
        let n = self.records.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let train_size = (n as f64 * train).round() as usize;
        let train_indices = &indices[0..train_size];
        let test_indices = &indices[train_size..];

        let train_records: Vec<Query> = train_indices
            .iter()
            .map(|&i| self.records[i].clone())
            .collect();
        let test_records: Vec<Query> = test_indices
            .iter()
            .map(|&i| self.records[i].clone())
            .collect();

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

pub struct EncryptedSample {
    pub class: Vec<LWE>, // one hot encoded class
    pub features: LUT,
}

impl EncryptedSample {
    fn one_hot_encode(class: &u64, n_classes: u64) -> Vec<u64> {
        let mut one_hot = vec![0; n_classes as usize];
        one_hot[*class as usize] = 1;
        one_hot
    }
    pub fn make_encrypted_sample(
        feature_vector: &Vec<u64>,
        class: &u64,
        n_classes: u64,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> Self {
        let feature_lut = LUT::from_vec(feature_vector, private_key, ctx);
        let one_hot_class = Self::one_hot_encode(class, n_classes);
        let class_lwes = one_hot_class
            .iter()
            .map(|x| private_key.allocate_and_encrypt_lwe(*x, ctx))
            .collect();
        Self {
            class: class_lwes,
            features: feature_lut,
        }
    }

    pub fn clone(&self) -> Self {
        Self {
            class: self.class.clone(),
            features: self.features.clone(),
        }
    }
}

pub struct EncryptedDatasetLut {
    pub records: Vec<EncryptedSample>,
    pub f: u64,
    pub n_classes: u64,
    pub features_domain: (u64, u64),
}

impl EncryptedDatasetLut {
    pub fn from_file(
        filepath: String,
        private_key: &PrivateKey,
        ctx: &mut Context,
        n_classes: u64,
    ) -> Self {
        let mut rdr = csv::Reader::from_path(filepath).unwrap();
        let mut records = Vec::new();
        let mut f = 0;
        let mut features_domain = (std::u64::MIN, std::u64::MAX);

        for result in rdr.records() {
            let record = result.unwrap();
            let mut label: u64 = 0;
            let mut record_vec = Vec::new();
            for (i, field) in record.iter().enumerate() {
                let value = field.parse::<u64>().unwrap();
                if i == record.len() - 1 {
                    label = value;
                } else {
                    record_vec.push(value);

                    if value < features_domain.0 {
                        features_domain.0 = value;
                    }
                    if value > features_domain.1 {
                        features_domain.1 = value;
                    }
                }
            }
            records.push(EncryptedSample::make_encrypted_sample(
                &record_vec,
                &label,
                n_classes,
                private_key,
                ctx,
            ));
            f = record_vec.len() as u64;
        }

        if f > ctx.full_message_modulus() as u64 {
            panic!("Number of features exceeds the modulus");
        }

        Self {
            records,
            f,
            n_classes,
            features_domain,
        }
    }

    pub fn split(&self, train: f64) -> (EncryptedDatasetLut, EncryptedDatasetLut) {
        let mut rng = rand::thread_rng();
        let n = self.records.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let train_size = (n as f64 * train).round() as usize;
        let train_indices = &indices[0..train_size];
        let test_indices = &indices[train_size..];

        let train_records: Vec<EncryptedSample> = train_indices
            .iter()
            .map(|&i| self.records[i].clone())
            .collect();
        let test_records: Vec<EncryptedSample> = test_indices
            .iter()
            .map(|&i| self.records[i].clone())
            .collect();

        (
            EncryptedDatasetLut {
                records: train_records,
                f: self.f,
                n_classes: self.n_classes,
                features_domain: self.features_domain,
            },
            EncryptedDatasetLut {
                records: test_records,
                f: self.f,
                n_classes: self.n_classes,
                features_domain: self.features_domain,
            },
        )
    }

    pub fn from_clear_dataset(
        dataset: &ClearDataset,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> Self {
        let records = dataset
            .records
            .iter()
            .map(|sample| {
                EncryptedSample::make_encrypted_sample(
                    &sample.features,
                    &sample.class,
                    dataset.n_classes,
                    private_key,
                    ctx,
                )
            })
            .collect();
        Self {
            records,
            f: dataset.f,
            n_classes: dataset.n_classes,
            features_domain: dataset.features_domain,
        }
    }
}
