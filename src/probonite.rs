use std::time::Instant;

// TFHE
use tfhe::core_crypto::prelude::*;

// REVOLUT
use revolut::*;

type LWE = LweCiphertext<Vec<u64>>;

use crate::model::*;

const DEBUG: bool = false;

pub struct Query {
    class: LWE,
    features: LUT,
}

impl Query {
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
}

pub fn next_accumulators(
    accumulators: &Vec<LWE>,
    selector_bit: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LWE> {
    let not_selector_bit = public_key.not_lwe(selector_bit, ctx);
    let mut nexts_accumulators = Vec::new();
    accumulators.iter().for_each(|lwe| {
        let mut accumulator_right = public_key.lwe_mul_encrypted_bit(lwe, selector_bit, ctx);
        let mut accumulator_left = public_key.lwe_mul_encrypted_bit(lwe, &not_selector_bit, ctx);
        public_key.bootstrap_lwe(&mut accumulator_right, ctx);
        public_key.bootstrap_lwe(&mut accumulator_left, ctx);
        nexts_accumulators.push(accumulator_right);
        nexts_accumulators.push(accumulator_left);
    });

    nexts_accumulators
}

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

pub fn blind_leaf_increment(
    leaves: &mut Vec<Leaf>,
    accumulators: &Vec<LWE>,
    sample_class: &LWE,
    public_key: &PublicKey,
    ctx: &Context,
) {
    for i in 0..leaves.len() {
        public_key.blind_array_increment(
            &mut leaves[i].counts,
            &sample_class,
            &accumulators[i],
            ctx,
        );
    }
}

pub fn probonite(tree: &mut Tree, query: &Query, public_key: &PublicKey, ctx: &Context) {
    // First stage

    let start = Instant::now();
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query.features, index as usize, ctx);
    let b = public_key.leq_scalar(&feature, threshold, ctx);
    let not_b = public_key.not_lwe(&b, ctx);
    let mut accumulators = vec![b, not_b];
    let end = Instant::now();
    println!("First stage: {:?}", end.duration_since(start));

    // Internal Stages
    for i in 0..tree.nodes.len() {
        let start = Instant::now();
        let (threshold, feature_index) =
            blind_node_selection(&tree.nodes[i], &accumulators, public_key, ctx);

        if DEBUG {
            let private_key = key(ctx.parameters());
            let t_selected = private_key.decrypt_lwe(&threshold, ctx);
            let f_selected = private_key.decrypt_lwe(&feature_index, ctx);
            println!("selected:({t_selected}, {f_selected})");
        }

        let feature = public_key.blind_array_access(&feature_index, &query.features, ctx);
        let b = public_key.blind_lt_bma_mv(&threshold, &feature, ctx);
        accumulators = next_accumulators(&accumulators, &b, public_key, ctx);
        let end = Instant::now();
        println!("Internal stage {}: {:?}", i, end.duration_since(start));
    }

    // Last stage : increment the leaves and get the majority class through argmax
    let start = Instant::now();
    blind_leaf_increment(
        &mut tree.leaves,
        &accumulators,
        &query.class,
        public_key,
        ctx,
    );
    let end = Instant::now();
    println!("Last stage: {:?}", end.duration_since(start));
}
