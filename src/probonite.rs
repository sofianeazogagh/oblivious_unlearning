// TFHE
use tfhe::core_crypto::prelude::*;

// REVOLUT
use revolut::*;

type LWE = LweCiphertext<Vec<u64>>;

struct Root {
    threshold: u64,
    feature_index: u64,
}

impl Root {
    pub fn print(&self) {
        println!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct InternalNode {
    threshold: u64,
    feature_index: u64,
}

impl InternalNode {
    pub fn print(&self) {
        print!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct Leaf {
    counts: LUT,
}

impl Leaf {
    pub fn print(&self, private_key: &PrivateKey, ctx: &Context, n_classes: u64) {
        // self.counts.print(private_key, ctx);
        let array = self.counts.to_array(private_key, ctx);
        print!("{:?}", &array[..n_classes as usize]);
    }
}

pub struct Tree {
    root: Root,
    nodes: Vec<Vec<InternalNode>>,
    leaves: Vec<Leaf>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            root: Root {
                threshold: 0,
                feature_index: 0,
            },
            nodes: Vec::new(),
            leaves: Vec::new(),
        }
    }

    pub fn generate_random_tree(depth: u64, n_classes: u64, ctx: &Context) -> Self {
        let mut tree = Self::new();

        // Generate the root
        tree.root.threshold = rand::random::<u64>() % ctx.full_message_modulus() as u64;
        tree.root.feature_index = rand::random::<u64>() % ctx.full_message_modulus() as u64;

        // Generate the nodes
        // Generate internal nodes for each level
        for _ in 1..depth {
            let mut stage = Vec::new();

            // Number of nodes at this level is 2^level_index
            let num_nodes = 2u64.pow(tree.nodes.len() as u32 + 1);

            for _ in 0..num_nodes {
                stage.push(InternalNode {
                    threshold: rand::random::<u64>() % ctx.full_message_modulus() as u64,
                    feature_index: rand::random::<u64>() % ctx.full_message_modulus() as u64,
                });
            }
            tree.nodes.push(stage);
        }

        // Generate the leaves
        let num_leaves = 2u64.pow(depth as u32);
        for _ in 0..num_leaves {
            let counts = LUT::from_vec_trivially(&vec![0u64; n_classes as usize], ctx);
            tree.leaves.push(Leaf { counts });
        }

        tree
    }

    pub fn print_tree(&self, private_key: &PrivateKey, ctx: &Context, n_classes: u64) {
        self.root.print();
        for (i, stage) in self.nodes.iter().enumerate() {
            for node in stage {
                node.print();
                print!(" ");
            }
            println!("");
        }
        for leaf in self.leaves.iter() {
            leaf.print(private_key, ctx, n_classes);
            print!(" ");
        }
        println!("");
    }
}

pub struct Query {
    class: LWE,
    features: LUT,
}

impl Query {
    pub fn new(
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
        let accumulator_right = public_key.lwe_mul_encrypted_bit(lwe, selector_bit, ctx);
        let accumulator_left = public_key.lwe_mul_encrypted_bit(lwe, &not_selector_bit, ctx);
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
    let index = tree.root.feature_index;
    let threshold = tree.root.threshold;
    let feature = public_key.lut_extract(&query.features, index as usize, ctx);
    let b = public_key.leq_scalar(&feature, threshold, ctx);
    let not_b = public_key.not_lwe(&b, ctx);
    let mut accumulators = vec![b, not_b];

    // Internal Stages
    for i in 0..tree.nodes.len() {
        let (threshold, feature_index) =
            blind_node_selection(&tree.nodes[i], &accumulators, public_key, ctx);

        let feature = public_key.blind_array_access(&feature_index, &query.features, ctx);
        let b = public_key.blind_lt_bma_mv(&threshold, &feature, ctx);
        accumulators = next_accumulators(&accumulators, &b, public_key, ctx);
    }

    // Last stage : increment the leaves and get the majority class through argmax
    blind_leaf_increment(
        &mut tree.leaves,
        &accumulators,
        &query.class,
        public_key,
        ctx,
    );
}
