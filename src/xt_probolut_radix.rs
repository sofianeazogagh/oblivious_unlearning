// DATASET
use crate::dataset::*;

// MODEL
use crate::model::*;
use crate::ClearDataset;
use crate::ClearTree;
use rayon::iter::IndexedParallelIterator;
// REVOLUT
use revolut::{key, Context, PrivateKey, PublicKey, LUT};
type LWE = LweCiphertext<Vec<u64>>;

use tfhe::boolean::prelude::Ciphertext;
use tfhe::integer::ciphertext::BaseRadixCiphertext;
use tfhe::integer::RadixCiphertext;
use tfhe::shortint::backward_compatibility::public_key;
// TFHE
use tfhe::{core_crypto::prelude::LweCiphertext, shortint::parameters::*};

// RAYON
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::iter::IntoParallelRefIterator;
use tfhe::integer::IntegerCiphertext;

use std::env::args;
// TIME
#[allow(unused_imports)]
use std::time::Instant;

// HELPERS
use crate::helpers::*;
use crate::{create_progress_bar, finish_progress, inc_progress, make_pb};

// CONSTANTS
use crate::VERBOSE;

pub struct XTForestLUT<'a> {
    pub trees: Vec<TreeLUT>,
    pub final_counts: Vec<Vec<Vec<RadixCiphertext>>>,
    pub n_classes: u64,
    pub depth: u64,
    pub public_key: PublicKey,
    pub private_key: PrivateKey,
    pub ctx: &'a Context,
}

impl<'a> XTForestLUT<'a> {
    pub fn from_clear_forest(forest: &Vec<ClearTree>, public_key: &PublicKey, ctx: &'a mut Context, private_key: &PrivateKey) -> Self {
        let mut trees = Vec::new();
        let n_classes = forest[0].n_classes;
        let depth = forest[0].depth;

        for tree in forest {
            trees.push(TreeLUT::from_clear_tree(tree, &ctx));
        }

        Self {
            trees,
            n_classes,
            depth,
            public_key: public_key.clone(),
            private_key: private_key.clone(),
            ctx,
            final_counts: Vec::new(),
        }
    }

    pub fn train_single_tree(
        &self,
        train_dataset: &EncryptedDatasetLut,
        tree: &TreeLUT,
        tree_idx: u64,
        mp: &MultiProgress,
    ) -> Vec<Vec<LUT>> {
        let tree_depth = tree.depth;
        let pb = make_pb(mp, train_dataset.records.len() as u64, tree_idx.to_string());

        let luts_samples = (0..train_dataset.records.len() as u64)
            .into_par_iter()
            .map(|j| {
                let result = self.probolut_for_training(
                    &tree,
                    &train_dataset.records[j as usize],
                    &self.public_key,
                    &self.ctx,
                );
                inc_progress!(&pb);
                result
            })
            .collect::<Vec<_>>();

        finish_progress!(pb);
        luts_samples
    }

    pub fn train(&mut self, train_dataset: &EncryptedDatasetLut, private_key: &PrivateKey) {
        let mp: MultiProgress = MultiProgress::new();
        let trained_trees_and_counts: Vec<(&TreeLUT, Vec<Vec<LUT>>)> = (0..self.trees.len())
            .into_par_iter()
            .map(|i| {
                let tree = &self.trees[i];
                let luts_samples = self.train_single_tree(train_dataset, tree, i as u64, &mp);
                // tree.print_tree(&private_key, &self.ctx);
                (tree, luts_samples)
            })
            .collect();


        let summed_counts_radix: Vec<Vec<Vec<RadixCiphertext>>> = trained_trees_and_counts
            .iter()
            .map(|(_, luts_samples)| self.aggregate_tree_counts_radix(luts_samples))
            .collect();

        
        let mut final_counts = Vec::new();
        for i in 0..summed_counts_radix.len() {
            let mut class_counts = Vec::new();
            for j in 0..summed_counts_radix[i].len() {
                let mut leaf_counts = Vec::new();
                for k in 0..summed_counts_radix[i][j].len() {
                    let dec = self.private_key
                    .decrypt_radix_ciphertext(&summed_counts_radix[i][j][k]);
                leaf_counts.push(dec);
            }
            class_counts.push(leaf_counts);
        }
        final_counts.push(class_counts);
    }
    // println!("Encrypted version : {:?}", final_counts);
    self.final_counts = summed_counts_radix;
    // println!("Shape of final counts: ({:?}, {:?}, {:?})", self.final_counts.len(), self.final_counts[0].len(), self.final_counts[0][0].len());
    }

    pub fn test(&mut self, test_dataset: &EncryptedDatasetLut, private_key: &PrivateKey) {
        let mp = MultiProgress::new();
        let pb = make_pb(&mp, test_dataset.records.len() as u64, "_");

        let mut correct = 0;
        let mut total = 0;

        let results:Vec<Vec<Vec<u64>>> = test_dataset.records.par_iter().enumerate().map(|(i, sample)| {
            inc_progress!(&pb);
            let mut sample_votes = Vec::new();
            let mut votes = vec![0; self.n_classes as usize];
            for (tree_index, tree) in self.trees.iter().enumerate() {
                // sample.print(private_key, &self.ctx, self.n_classes);
                let leaf =
                    self.probolut_inference(tree, &sample.features, &self.public_key, &self.ctx);
                let mut dec_tree_counts = Vec::new();
                let count_vecs = self.get_counts_from_selector(&leaf, tree_index as u64);
                count_vecs.iter().for_each(|count_vec|{
                    let prediction = private_key.decrypt_radix_ciphertext(count_vec);
                    dec_tree_counts.push(prediction);
                });
                sample_votes.push(dec_tree_counts);
            };
            sample_votes
        }).collect();
        
        finish_progress!(pb);


        // println!("Results of the test set: {:?}", results);

        results.iter().enumerate().for_each(|(i, sample_votes) |{
            let mut agg_sample_votes = vec![0; self.n_classes as usize];

            (0..self.trees.len()).for_each(|tree_index|{
                (0..self.n_classes).for_each(|class_index|{
                    agg_sample_votes[class_index as usize] += sample_votes[tree_index][class_index as usize];
                });
            });
            let predicted_label = agg_sample_votes.iter().enumerate().max_by_key(|x| x.1).unwrap().0;
            // println!("Predicted label: {:?}", predicted_label);
            let mut true_label = 0;
            test_dataset.records[i].class.iter().enumerate().for_each(|(j, lwe)|{
                let dec = self.private_key.decrypt_lwe(&lwe, &self.ctx);
                if dec == 1 {
                  true_label = j;
                //   println!("True label: {:?}", true_label);
                }
            });
            if predicted_label == true_label {
                correct += 1;
            }
            total += 1;
        });

        println!("Accuracy: {}", correct as f64 / total as f64);
    }


    fn get_counts_from_selector(&self, selector: &LWE, tree_index: u64) -> Vec<BaseRadixCiphertext<tfhe::shortint::Ciphertext>>{
        let mut counts:Vec<BaseRadixCiphertext<tfhe::shortint::Ciphertext>> = Vec::new();
        let n_leaves = 2u32.pow(self.depth as u32);
        let mut lut0 = LUT::from_vec_trivially(&vec![0; n_leaves as usize], &self.ctx);
        let lwe1 = self.public_key.allocate_and_trivially_encrypt_lwe(1, &self.ctx);
        let lwe0 = self.public_key.allocate_and_trivially_encrypt_lwe(0, &self.ctx);
        self.public_key.blind_array_increment(&mut lut0, &selector, &lwe1, &self.ctx);
        let lwe_selectors:Vec<LWE> = (0..n_leaves).map(|leaf_index|{
            self.public_key.lut_extract(&lut0, leaf_index as usize, &self.ctx)
        }).collect();

        // lwe_selectors.clone().into_iter().for_each(|selector|{
        //     self.private_key.debug_lwe("selector", &selector, &self.ctx);
        // });

        let shortin_0 = self.public_key.to_shortint_ciphertext(lwe0.clone(), &self.ctx);


        let selected_counts = (0..self.n_classes).map(|class_index|{
            let mut integer_class_acc = BaseRadixCiphertext::from_blocks(vec![shortin_0.clone();3]);
            let counts = (0..n_leaves).for_each(|leaf_index|{
                let shortint_selector = self.public_key.to_shortint_ciphertext(lwe_selectors[leaf_index as usize].clone(), &self.ctx);
                let blocks = vec![shortint_selector.clone(), shortin_0.clone(), shortin_0.clone()];
                let mut integer_selector = BaseRadixCiphertext::from_blocks(blocks);
                let mut integer_selector_copy = integer_selector.clone();
                let decrypted_integer = self.private_key.decrypt_radix_ciphertext(&integer_selector);
                // println!("Decrypted integer: {:?}", decrypted_integer);
                self.public_key.integer_key.mul_assign_parallelized(&mut integer_selector_copy, &self.final_counts[tree_index as usize][class_index as usize][leaf_index as usize]);
                self.public_key.integer_key.add_assign_parallelized(&mut integer_class_acc, &integer_selector_copy);
            });
            integer_class_acc
        }).collect();
        selected_counts
    }



fn aggregate_tree_counts_radix(
    &self,
    luts_samples: &Vec<Vec<LUT>>,
) -> Vec<Vec<RadixCiphertext>> {
    let n_samples = luts_samples.len() as u64;
    let n_classes = luts_samples[0].len() as u64;
    let n_leaves = 2u32.pow(self.depth as u32);

    let output: Vec<Vec<Vec<LWE>>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            (0..n_classes)
                .map(|j| luts_samples[i as usize][j as usize].to_many_lwe(&self.public_key, &self.ctx))
                .collect()
        })
        .collect();

    (0..n_classes)
        .into_par_iter()
        .map(|j| {
            (0..n_leaves)
                .map(|k| {
                    let counts_to_sum: Vec<_> = (0..n_samples)
                        .map(|l| output[l as usize][j as usize][k as usize].clone())
                        .collect();

                    self.public_key.lwe_add_into_radix_ciphertext(counts_to_sum, 1, &self.ctx)
                })
                .collect()
        })
        .collect()
}


    fn probolut_for_training(
        &self,
        tree: &TreeLUT,
        sample: &EncryptedSample,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> Vec<LUT> {
        let leaf = self.probolut_inference(tree, &sample.features, public_key, ctx);

        let mut counts = tree.leaves.clone();
        for c in 0..tree.n_classes {
            public_key.blind_array_increment(
                &mut counts[c as usize],
                &leaf,
                &sample.class[c as usize],
                ctx,
            );
        }

        counts
    }

    fn probolut_inference(
        &self,
        tree: &TreeLUT,
        query: &LUT,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> LWE {
        // First stage
        let index = tree.root.feature_index;
        let threshold = tree.root.threshold;
        let feature = public_key.lut_extract(&query, index as usize, ctx);
        let b = public_key.lt_scalar(&feature, threshold, ctx);
        // private_key.debug_lwe("b", &b, ctx);

        // Internal Stages
        let mut selector = b.clone();
        // private_key.debug_lwe("selector", &selector, ctx);
        for i in 0..tree.stages.len() {
            let (lut_index, lut_threshold) = &tree.stages[i];
            let feature_index = public_key.blind_array_access(&selector, &lut_index, ctx);
            let threshold = public_key.blind_array_access(&selector, &lut_threshold, ctx);
            let feature = public_key.blind_array_access(&feature_index, &query, ctx);
            let b = public_key.blind_lt_bma_mv(&feature, &threshold, ctx);
            selector = public_key.lwe_mul_add(&b, &selector, 2);
            // private_key.debug_lwe("b", &b, ctx);
            // private_key.debug_lwe("selector_updated", &selector, ctx);
        }
        selector
    }


}

pub fn train_test_probolut_vs_clear(args: Args) {
    // const TREE_DEPTH: u64 = 4;
    // const DATASET: &str = "iris";
    const N_CLASSES: u64 = 3;
    // const NUM_EXPERIMENTS: u64 = 10;

    // let precisions = vec![2, 3, 4, 5];
    // let num_forests = vec![10, 40, 60];
    // let quantizations = vec!["uni", "non_uni", "fd"];

    let precisions = args.precisions;
    let num_forests = args.num_forests;
    let quantizations = args.quantizations;
    let DATASET = args.dataset_name;
    let depths = args.depths;
    let NUM_EXPERIMENTS = args.number_of_experiments;

    for q in quantizations {
        for b in &precisions {
            for m in &num_forests {
                for d in depths.clone() {
                    let TREE_DEPTH = d;
                    let dataset_name = if q == "" {
                        format!("{}_{}bits", DATASET, b)
                    } else {
                        format!("{}_{}_{}bits", DATASET, q, b)
                    };
                    let M = *m;

                    println!("\n SUMMARY: {}, {} trees ", dataset_name, *m);
                    println!("----------------------------------------");

                    let mut ctx = if *b <= 4 {
                        Context::from(PARAM_MESSAGE_4_CARRY_0)
                    } else {
                        Context::from(PARAM_MESSAGE_5_CARRY_0)
                    };
                    let private_key = key(ctx.parameters());
                    let public_key = &private_key.public_key;

                    for _ in 0..NUM_EXPERIMENTS {
                        let start = Instant::now();
                        let clear_dataset =
                            ClearDataset::from_file("data/".to_string() + &dataset_name + ".csv");
                        let (train_dataset, test_dataset) = clear_dataset.split(0.8);

                        if VERBOSE {
                            println!("\n --------- Training the clear forest ---------");
                        }
                        let mp = MultiProgress::new();
                        let mut clear_forest_trained = (0..M)
                            .into_par_iter()
                            .map(|i| {
                                let mut clear_tree = ClearTree::generate_clear_random_tree(
                                    TREE_DEPTH,
                                    clear_dataset.n_classes,
                                    clear_dataset.max_features,
                                    clear_dataset.f,
                                );

                                
                                let pb =
                                make_pb(&mp, train_dataset.records.len() as u64, i.to_string());
                                
                                train_dataset.records.iter().for_each(|sample| {
                                    clear_tree.update_statistic(sample);
                                    inc_progress!(&pb);
                                });
                                
                                
                                // clear_tree.print();

                                finish_progress!(pb);
                                clear_tree
                            })
                            .collect::<Vec<_>>();

                        let training_time = Instant::now() - start;
                        let start = Instant::now();

                        if VERBOSE {
                            println!("\n --------- Testing the clear forest ---------");
                        }
                        let mp = MultiProgress::new();
                        let pb = make_pb(&mp, test_dataset.records.len() as u64, "_");

                        let mut correct = 0;
                        let mut total = 0;
                        test_dataset.records.iter().for_each(|sample| {
                            let mut votes = vec![0; N_CLASSES as usize];
                            clear_forest_trained.iter_mut().for_each(|tree| {
                                let leaf = tree.infer(sample);
                                for c in 0..N_CLASSES {
                                    votes[c as usize] += leaf.counts[c as usize];
                                }
                            });

                            let (predicted_label, _) =
                                votes.iter().enumerate().max_by_key(|x| x.1).unwrap();
                            let true_label = sample.class;
                            if predicted_label == true_label as usize {
                                correct += 1;
                            }
                            total += 1;
                            inc_progress!(&pb);
                        });

                        let testing_time = Instant::now() - start;
                        finish_progress!(pb);

                        println!("\n-------- Forest on clear data - Accuracy -------- ");
                        println!("Correct: {}, Total: {}", correct, total);
                        let accuracy = correct as f64 / total as f64;
                        println!("\n Accuracy: {} ", accuracy);

                        log(
                            &format!("logs/{}_{}d_{}m_clear.csv", dataset_name, TREE_DEPTH, M),
                            &format!(
                                "{},{},{}",
                                accuracy,
                                training_time.as_millis(),
                                testing_time.as_millis()
                            ),
                        );

                        if VERBOSE {
                            println!("\n --------- Training the forest on private data ---------");
                        }

                        let enc_train_dataset = EncryptedDatasetLut::from_clear_dataset(
                            &train_dataset,
                            &private_key,
                            &mut ctx,
                        );
                        let enc_test_dataset = EncryptedDatasetLut::from_clear_dataset(
                            &test_dataset,
                            &private_key,
                            &mut ctx,
                        );
                        // Train forest

                        let start = Instant::now();
                        let mp = MultiProgress::new();
                        let mut xt_forest = XTForestLUT::from_clear_forest(&clear_forest_trained, public_key, &mut ctx, &private_key);

                        xt_forest.train(&enc_train_dataset, &private_key);

                        xt_forest.test(&enc_test_dataset, &private_key);


                        // let testing_time = Instant::now() - start;

                        // println!("\n-------- Forest on encrypted data - Accuracy -------- ");
                        // println!("Correct: {}, Total: {}", correct, total);
                        // let accuracy = correct as f64 / total as f64;
                        // println!("\n Accuracy: {} ", accuracy);

                        // log(
                        //     &format!(
                        //         "logs_1st_campaign/{}_{}d_{}m_probolut.csv",
                        //         dataset_name, TREE_DEPTH, M
                        //     ),
                        //     &format!(
                        //         "{},{},{}",
                        //         accuracy,
                        //         training_time.as_millis(),
                        //         testing_time.as_millis()
                        //     ),
                        // );
                    }
                }
            }
        }
    }
}

