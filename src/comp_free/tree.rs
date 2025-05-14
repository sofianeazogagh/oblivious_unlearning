use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::iter::IntoParallelRefIterator;
use revolut::radix::NyblByteLUT;
use serde_json::json;
use serde_json::Value;
use tfhe::boolean::public_key;

use super::ctree::*;
use super::dataset::*;
use super::RLWE;

use crate::comp_free::{DEBUG, OBLIVIOUS};
use crate::*;

// - Tree
// 	- root : (index, threshold)
// 	- Stages : Vec<>
// 	- Leaves : Vec<Vec<ByteLWE>>
// 	- function compile (sample : EncryptedSample) :-> CTree

pub struct Node {
    pub threshold: u64,
    pub index: u64,
}

#[derive(Clone)]
// pub struct Leaf {
//     pub classes: Vec<ByteLWE>,
// }

pub struct Classes {
    pub class: NyblByteLUT,
}

pub struct Tree {
    pub root: Node,
    pub stages: Vec<Vec<Node>>, // Stages[i] is the i-th stages of the tree
    // pub leaves: Vec<Leaf>,
    pub leaves_lut: Vec<Classes>,
    pub final_leaves: Vec<LWE>,
    pub depth: u64,
    pub n_classes: u64,
}

#[allow(dead_code)]
impl Tree {
    pub fn new(depth: u64, n_classes: u64) -> Self {
        Self {
            root: Node {
                threshold: 0,
                index: 0,
            },
            stages: Vec::new(),
            // leaves: Vec::new(),
            leaves_lut: Vec::new(),
            depth,
            n_classes,
            final_leaves: Vec::new(),
        }
    }

    pub fn new_random_tree(
        depth: u64,
        n_classes: u64,
        f: u64,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> Self {
        let mut tree = Self::new(depth, n_classes);

        // Generate the root
        let N = ctx.polynomial_size().0 as u64;
        tree.root.threshold = rand::random::<u64>() % N;
        tree.root.index = rand::random::<u64>() % f;

        // Generate the nodes
        for level_index in 1..depth {
            // Number of nodes at this level is 2^level_index
            let num_nodes = 2u64.pow(level_index as u32);

            let mut stage = Vec::new();
            for _ in 0..num_nodes {
                stage.push(Node {
                    threshold: rand::random::<u64>() % N,
                    index: rand::random::<u64>() % f,
                });
            }
            tree.stages.push(stage);
        }

        // Option 2 : The leaf are horizontally packed into a NyblByteLUT (one per class)
        let num_classes = tree.n_classes;
        for _ in 0..num_classes {
            let class = NyblByteLUT::from_bytes_trivially(&[0x00; 16], ctx);
            tree.leaves_lut.push(Classes { class });
        }

        tree
    }

    pub fn new_random_tree_with_seed(
        depth: u64,
        n_classes: u64,
        f: u64,
        public_key: &PublicKey,
        ctx: &Context,
        seed: u64,
    ) -> Self {
        let mut tree = Self::new(depth, n_classes);

        // Create a seeded random number generator
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate the root
        let N = ctx.polynomial_size().0 as u64;
        tree.root.threshold = rng.gen::<u64>() % N;
        tree.root.index = rng.gen::<u64>() % f;

        // Generate the nodes
        for level_index in 1..depth {
            // Number of nodes at this level is 2^level_index
            let num_nodes = 2u64.pow(level_index as u32);

            let mut stage = Vec::new();
            for _ in 0..num_nodes {
                stage.push(Node {
                    threshold: rng.gen::<u64>() % N,
                    index: rng.gen::<u64>() % f,
                });
            }
            tree.stages.push(stage);
        }

        // Option 2 : The leaf are horizontally packed into a NyblByteLUT (one per class)
        let num_classes = tree.n_classes;
        for _ in 0..num_classes {
            let class = NyblByteLUT::from_bytes_trivially(&[0x00; 16], ctx);
            tree.leaves_lut.push(Classes { class });
        }

        tree
    }

    pub fn leaves_update(
        &mut self,
        selector: &LWE,
        class: &[LWE],
        public_key: &PublicKey,
        ctx: &Context,
    ) {
        if !OBLIVIOUS {
            for i in 0..self.n_classes {
                self.leaves_lut[i as usize].class.blind_array_maybe_inc(
                    &selector,
                    &class[i as usize],
                    ctx,
                    public_key,
                );
            }
        } else {
            for i in 0..self.n_classes {
                self.leaves_lut[i as usize]
                    .class
                    .blind_array_maybe_inc_or_dec(&selector, &class[i as usize], ctx, public_key);
            }
        }
    }

    // Function to do at the end of the training (using the LUTs version)
    pub fn leaves_majority(&mut self, public_key: &PublicKey, ctx: &Context) {
        // Unpack the LUTs into the Vec of Vec<ByteLWE> : gives n_classes lines of 2^depth ByteLWE
        let mut leaves = Vec::new();
        for i in 0..self.n_classes {
            leaves.push(
                self.leaves_lut[i as usize]
                    .class
                    .to_many_blwes(public_key, ctx),
            );
        }

        // Make each column as a Vec<ByteLWE> : gives 2^depth columns of n_classes ByteLWE
        let mut leaves_columns = Vec::new();
        for i in 0..leaves[0].len() {
            let mut column = Vec::new();
            for leaf in leaves.iter() {
                column.push(leaf[i as usize].clone());
            }
            leaves_columns.push(column);
        }

        // Compute the majority of each leaf
        let mut majority = Vec::new();
        let zero_blwe = ByteLWE::from_byte_trivially(0x00, ctx, public_key);
        for column in leaves_columns.iter_mut() {
            // We add a zero for the extra class of abstention (then if the counts are [0,0,0] it becomes [0,0,0,0] and the argmax is 3 instead of 2)
            column.push(zero_blwe.clone());
            let maj = public_key.blind_argmax_byte_lwe(&column, ctx);
            majority.push(maj.lo); // We take the lo part of the ByteLWE since the classes are < 16
        }

        self.final_leaves = majority;
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        for sample in dataset.records.iter() {
            let start = Instant::now();
            // Compile the tree
            let ctree = CTree::new(self, &sample.features, public_key, ctx);
            let duration = start.elapsed();
            println!("[TIME] Compile tree: {:?}", duration);

            let start = Instant::now();
            // Traverse the tree
            let selector = ctree.evaluate(public_key, ctx);
            let duration = start.elapsed();
            println!("[TIME] Evaluate tree: {:?}", duration);

            if DEBUG {
                let private_key = key(ctx.parameters());
                let start = Instant::now();
                ctree.print(&private_key, ctx);
                let duration = start.elapsed();
                println!("[TIME] Print tree: {:?}", duration);
                println!("Selector: {}", private_key.decrypt_lwe(&selector, ctx));
            }

            // Update the leaves
            let start = Instant::now();
            self.leaves_update(&selector, &sample.class, public_key, ctx);
            let duration = start.elapsed();
            println!("[TIME] Leaves update: {:?}", duration);

            if DEBUG {
                let private_key = key(ctx.parameters());

                let decrypted_sample: Vec<Vec<u64>> = sample
                    .features
                    .iter()
                    .map(|glwe| private_key.decrypt_and_decode_glwe(glwe, ctx))
                    .collect();
                let mut clear_sample = Vec::new();
                for vector in decrypted_sample.iter() {
                    let sum: u64 = vector.iter().sum();
                    clear_sample.push(ctx.polynomial_size().0 as u64 - sum);
                }
                println!("[FHE] Sample: {:?}", clear_sample);
                println!("Tree: ");
                self.print_tree(&private_key, ctx);
            }
        }
    }

    pub fn test(&self, sample_features: &Vec<RLWE>, public_key: &PublicKey, ctx: &Context) -> LWE {
        let ctree = CTree::new(self, sample_features, public_key, ctx);
        let selector = ctree.evaluate(public_key, ctx);

        if DEBUG {
            let private_key = key(ctx.parameters());
            let decrypted_sample: Vec<Vec<u64>> = sample_features
                .iter()
                .map(|glwe| private_key.decrypt_and_decode_glwe(glwe, ctx))
                .collect();
            let mut clear_sample = Vec::new();
            for vector in decrypted_sample.iter() {
                let sum: u64 = vector.iter().sum();
                clear_sample.push(ctx.polynomial_size().0 as u64 - sum);
            }
            println!("[FHE] Sample: {:?}", clear_sample);
            ctree.print(&private_key, ctx);
            println!("Selector: {}", private_key.decrypt_lwe(&selector, ctx));
        }

        let lut_leaves = LUT::from_vec_of_lwe(&self.final_leaves, public_key, ctx);
        public_key.blind_array_access(&selector, &lut_leaves, ctx)
    }

    // Function to print the tree (using the LUTs version)
    pub fn print_tree(&self, private_key: &PrivateKey, ctx: &Context) {
        println!("-----------[(t,f)]-----------");
        println!("Root: ({},{})", self.root.threshold, self.root.index);
        for stage in self.stages.iter() {
            for node in stage.iter() {
                print!("({},{}) ", node.threshold, node.index);
            }
            println!("");
        }
        println!("");
        for leaf in self.leaves_lut.iter() {
            leaf.class
                .print_bytes(&private_key.public_key, private_key, ctx);
        }
        println!("-----------------------------");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp_free::*;
    use revolut::{Context, PrivateKey, PublicKey};

    #[test]
    fn test_new_tree_and_print() {
        // Create a context
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Generate public and private keys
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // Define parameters for the new tree
        let depth = 2;
        let n_classes = 2;
        let f = ctx.parameters().polynomial_size.0 as u64;

        // Create a new tree
        let tree = Tree::new_random_tree_with_seed(depth, n_classes, f, &public_key, &ctx, 1);

        // Print the tree
        tree.print_tree(&private_key, &ctx);
    }

    #[test]
    fn test_compile_tree() {
        // Create a context
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Generate public and private keys
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // Model parameters
        let depth = 4;
        let n_classes = 3;
        // let f = ctx.parameters().polynomial_size.0 as u64;
        let f = ctx.full_message_modulus() as u64;

        // New Tree
        let mut tree = Tree::new_random_tree(depth, n_classes, f, &public_key, &ctx);
        println!("Tree: ");
        tree.print_tree(&private_key, &ctx);

        // New Sample
        let sample_vector = (0..f).collect();
        let class = 1;
        let encrypted_sample = EncryptedSample::make_encrypted_sample(
            &sample_vector,
            &class,
            3,
            &private_key,
            &mut ctx,
        );

        println!("Encrypted Sample: ");
        encrypted_sample.print(&private_key, &ctx);

        // Compile the tree
        let ctree = CTree::new(&tree, &encrypted_sample.features, &public_key, &ctx);
        println!("CTree: ");
        ctree.print(&private_key, &ctx);

        // Evaluate the tree
        let selector = ctree.evaluate(&public_key, &ctx);
        println!("Selector: {}", private_key.decrypt_lwe(&selector, &ctx));
    }

    #[test]
    fn test_leaves_update_one_sample() {
        // Create a context
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Generate public and private keys
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // Model parameters
        let depth = 4;
        let n_classes = 3;
        // let f = ctx.parameters().polynomial_size.0 as u64;
        let f = ctx.full_message_modulus() as u64;

        // New Tree
        let mut tree = Tree::new_random_tree(depth, n_classes, f, &public_key, &ctx);
        println!("Tree: ");
        tree.print_tree(&private_key, &ctx);

        // New Sample
        let sample_vector = (0..f).collect();
        let class = 1;
        let encrypted_sample = EncryptedSample::make_encrypted_sample(
            &sample_vector,
            &class,
            n_classes,
            &private_key,
            &mut ctx,
        );

        println!("Encrypted Sample: ");
        encrypted_sample.print(&private_key, &ctx);

        // Compile the tree
        let ctree = CTree::new(&tree, &encrypted_sample.features, &public_key, &ctx);
        println!("CTree: ");
        ctree.print(&private_key, &ctx);

        // Evaluate the tree
        let selector = ctree.evaluate(&public_key, &ctx);
        println!("Selector: {}", private_key.decrypt_lwe(&selector, &ctx));

        // Update the leaves
        let start = Instant::now();
        tree.leaves_update(&selector, &encrypted_sample.class, &public_key, &ctx);
        let duration = start.elapsed();
        println!("Time taken: {:?}", duration);

        println!("Updated Tree: ");
        tree.print_tree(&private_key, &ctx);
    }

    #[test]
    fn test_leaves_update_multiple_samples() {
        // Create a context
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Generate public and private keys
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // Model parameters
        let depth = 4;
        let n_classes = 3;
        let f = ctx.full_message_modulus() as u64;

        // New Tree
        let mut tree = Tree::new_random_tree(depth, n_classes, f, &public_key, &ctx);
        println!("Initial Tree: ");
        tree.print_tree(&private_key, &ctx);

        // Process multiple samples
        let n_samples = 254;
        let mut samples = Vec::new();
        for sample_index in 0..n_samples {
            // New Sample
            let sample_vector = (0..f).collect();
            // let class = sample_index % n_classes; // Example: cycle through class labels
            let class = 1;
            let encrypted_sample = EncryptedSample::make_encrypted_sample(
                &sample_vector,
                &class,
                n_classes,
                &private_key,
                &mut ctx,
            );
            samples.push(encrypted_sample);
        }

        let start = Instant::now();
        for sample in samples {
            // Compile the tree
            let ctree = CTree::new(&tree, &sample.features, &public_key, &ctx);
            // Evaluate the tree
            let selector = ctree.evaluate(&public_key, &ctx);
            // Update the leaves
            tree.leaves_update(&selector, &sample.class, &public_key, &ctx);
        }
        let duration = start.elapsed();
        println!("Time taken: {:?}", duration);

        println!("Updated Tree after all samples: ");
        tree.print_tree(&private_key, &ctx);
    }
}
