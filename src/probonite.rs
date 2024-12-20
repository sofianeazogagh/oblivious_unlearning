#[allow(unused_imports)]
use std::time::Instant;

#[allow(unused_imports)]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[allow(unused_imports)]
use crate::dataset::*;
// TFHE
#[allow(unused_imports)]
use tfhe::core_crypto::prelude::*;

// REVOLUT
#[allow(unused_imports)]
use revolut::*;
type LWE = LweCiphertext<Vec<u64>>;

#[allow(unused_imports)]
use crate::model::*;

const DEBUG: bool = false;

pub struct Query {
    pub class: LWE,
    pub features: LUT,
}

impl Query {
    #[allow(dead_code)]
    pub fn make_query(
        feature_vector: &Vec<u64>,
        class: &u64,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> Self {
        let feature_lut = LUT::from_vec(feature_vector, private_key, ctx);
        let class_lwe = private_key.allocate_and_encrypt_lwe(class.clone(), ctx);
        Self {
            class: class_lwe,
            features: feature_lut,
        }
    }

    #[allow(dead_code)]
    pub fn clone(&self) -> Self {
        Self {
            class: self.class.clone(),
            features: self.features.clone(),
        }
    }
}

#[allow(dead_code)]
pub fn next_accumulators(
    accumulators: &Vec<LWE>,
    selector_bit: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LWE> {
    let not_selector_bit = public_key.not_lwe(selector_bit, ctx);
    let mut nexts_accumulators = Vec::new();
    accumulators.iter().for_each(|lwe| {
        let accumulator_left = public_key.lwe_mul_encrypted_bit(lwe, &not_selector_bit, ctx);
        let accumulator_right = public_key.lwe_mul_encrypted_bit(lwe, selector_bit, ctx);
        nexts_accumulators.push(accumulator_left);
        nexts_accumulators.push(accumulator_right);
    });

    nexts_accumulators
}

#[allow(dead_code)]
pub fn blind_node_selection(
    nodes: &Vec<InternalNode>,
    accumulators: &Vec<LWE>,
    public_key: &PublicKey,
    ctx: &Context,
) -> (LWE, LWE) {
    let mut thresholds = Vec::new();
    nodes.iter().for_each(|node| {
        thresholds.push(node.threshold);
    });
    let selected_threshold = public_key.private_selection(&thresholds, accumulators, ctx);

    let mut feature_indices = Vec::new();
    nodes.iter().for_each(|node| {
        feature_indices.push(node.feature_index);
    });
    let selected_feature_index = public_key.private_selection(&feature_indices, accumulators, ctx);

    (selected_threshold, selected_feature_index)
}

#[allow(dead_code)]
pub fn blind_leaf_increment(
    accumulators: &Vec<LWE>,
    sample_class: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LUT> {
    let luts = (0..accumulators.len())
        .into_par_iter()
        .map(|i| {
            let mut lut = LUT::from_lwe(&accumulators[i], public_key, ctx);
            public_key.blind_rotation_assign(
                &public_key.neg_lwe(&sample_class, ctx),
                &mut lut,
                ctx,
            );
            // lut = [0, 0, ..., 0, 1, 0, ..., 0] if acc[i] = 1
            // lut = [0, 0, ..., 0, 0, 0, ..., 0] if acc[i] = 0
            lut
        })
        .collect();

    luts
}

#[allow(dead_code)]
pub fn probonite(tree: &Tree, query: &Query, public_key: &PublicKey, ctx: &Context) -> Vec<LUT> {
    // Normal Probonite inference

    let accumulators = probonite_inference(tree, query, public_key, ctx);

    // Writing the class in the selected leaf
    let start = Instant::now();
    let luts = blind_leaf_increment(&accumulators.0, &query.class, public_key, ctx);
    let end = Instant::now();
    if DEBUG {
        println!("Last stage: {:?}", end.duration_since(start));
    }

    luts
}

#[allow(dead_code)]
pub fn probonite_inference(
    tree: &Tree,
    query: &Query,
    public_key: &PublicKey,
    ctx: &Context,
) -> (Vec<LWE>, f64) {
    let mut inference_time = 0.0;
    let start = Instant::now();
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query.features, index as usize, ctx);
    let b = public_key.leq_scalar(&feature, threshold, ctx);
    let not_b = public_key.not_lwe(&b, ctx);
    let mut accumulators = vec![b, not_b];
    let end = Instant::now();
    inference_time += end.duration_since(start).as_secs_f64();
    if DEBUG {
        println!("First stage: {:?}", end.duration_since(start));
    }

    // Internal Stages
    for i in 0..tree.nodes.len() {
        let start = Instant::now();
        let (threshold, feature_index) =
            blind_node_selection(&tree.nodes[i], &accumulators, public_key, ctx);
        let feature = public_key.blind_array_access(&feature_index, &query.features, ctx);
        let b = public_key.blind_lt_bma_mv(&threshold, &feature, ctx);

        if DEBUG {
            let private_key = key(ctx.parameters());
            let t_selected = private_key.decrypt_lwe(&threshold, ctx);
            let f_selected = private_key.decrypt_lwe(&feature_index, ctx);
            println!("selected:({t_selected}, {f_selected})");
            let feature_decrypted = private_key.decrypt_lwe(&feature, ctx);
            let threshold_decrypted = private_key.decrypt_lwe(&threshold, ctx);
            println!(
                "expected b: {}",
                (threshold_decrypted < feature_decrypted) as u64
            );
            private_key.debug_lwe("actual b : ", &b, ctx);
        }
        accumulators = next_accumulators(&accumulators, &b, public_key, ctx);
        let end = Instant::now();
        if DEBUG {
            println!("Internal stage {}: {:?}", i, end.duration_since(start));
        }
        inference_time += end.duration_since(start).as_secs_f64();
    }

    if DEBUG {
        println!("Total inference time: {:?}", inference_time);
    }

    (accumulators, inference_time)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tfhe::shortint::parameters::PARAM_MESSAGE_5_CARRY_0;

    #[test]
    fn test_probonite() {
        // Initialize context
        let mut ctx = Context::from(PARAM_MESSAGE_5_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        const TREE_DEPTH: u64 = 5;
        const N_CLASSES: u64 = 3;
        let f = ctx.full_message_modulus() as u64;

        // Generate a random tree
        let mut tree = Tree::generate_random_tree(TREE_DEPTH, N_CLASSES, f, &ctx);

        // Create a query with test data
        let feature_vector = vec![31; ctx.full_message_modulus() as usize];
        let features_vector = LUT::from_vec(&feature_vector, &private_key, &mut ctx);
        let class = private_key.allocate_and_encrypt_lwe(0, &mut ctx);

        let query = Query {
            features: features_vector,
            class: class,
        };

        // Run probonite
        let start = Instant::now();
        let counts = probonite(&tree, &query, &public_key, &ctx);
        let end = Instant::now();
        println!("Time probonite taken: {:?}", end.duration_since(start));

        let luts_samples = vec![counts];

        tree.sum_samples_luts_counts(&luts_samples, &public_key);

        tree.print_tree(&private_key, &ctx);
    }
}
