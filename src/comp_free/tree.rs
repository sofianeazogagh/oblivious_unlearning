use revolut::radix::NyblByteLUT;
use tfhe::boolean::public_key;

use super::ctree::*;
use super::dataset::*;

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
pub struct Leaf {
    pub classes: Vec<ByteLWE>,
}

pub struct Tree {
    pub root: Node,
    pub stages: Vec<Vec<Node>>, // Stages[i] is the i-th stages of the tree
    pub leaves: Vec<Leaf>,
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
            leaves: Vec::new(),
            depth,
            n_classes,
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
        tree.root.threshold = rand::random::<u64>() % f as u64;
        tree.root.index = rand::random::<u64>() % f;

        // Generate the nodes
        for level_index in 1..depth {
            // Number of nodes at this level is 2^level_index
            let num_nodes = 2u64.pow(level_index as u32);

            let mut stage = Vec::new();
            for _ in 0..num_nodes {
                stage.push(Node {
                    threshold: rand::random::<u64>() % f as u64,
                    index: rand::random::<u64>() % f,
                });
            }
            tree.stages.push(stage);
        }

        // Initialize the leaves to 0
        let num_leaves = 2u64.pow(depth as u32);
        for _ in 0..num_leaves {
            let leaf = Leaf {
                classes: vec![
                    ByteLWE::from_byte_trivially(0x00, ctx, public_key);
                    n_classes as usize
                ],
            };
            tree.leaves.push(leaf);
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
        // First convert the vector class to a vector of ByteLWE
        let byte_class: Vec<ByteLWE> = class
            .iter()
            .map(|c| ByteLWE {
                lo: c.clone(),
                hi: public_key.allocate_and_trivially_encrypt_lwe(0, ctx),
            })
            .collect();

        // Pack the leaves into a NyblByteLUT
        let leaves = self.leaves.clone();
        let mut nbluts = Vec::new();
        for i in 0..self.n_classes {
            let mut collected_classes = Vec::new();
            for leaf in &leaves {
                collected_classes.push(leaf.classes[i as usize].clone());
            }
            let nblut = NyblByteLUT::from_vec_of_blwes(&collected_classes, public_key, ctx);
            nbluts.push(nblut);
        }

        // Blind increment each NyblByteLUTs at the position of the selector with the class
        // TODO : Could be improved since we know that the value added is 0 or 1.
        // TODO : Or we could add a bunch of LWE (i.e a bunch of samples < 16) then convert it to a ByteLWE.
        for i in 0..self.n_classes {
            nbluts[i as usize].blind_array_add(&selector, &byte_class[i as usize], ctx, public_key);
        }

        // The new leaves are the Unpacked NyblByteLUTs

        //- Unpack the NyblByteLUTs into the leaves
        let mut unpacked_nbluts = Vec::new();
        for i in 0..nbluts.len() {
            unpacked_nbluts.push(nbluts[i as usize].to_many_blwes(public_key, ctx));
        }

        //- Update the leaves
        let p = unpacked_nbluts[0].len();
        for j in 0..p {
            let mut new_leaf = Vec::new();
            for i in 0..self.n_classes {
                new_leaf.push(unpacked_nbluts[i as usize][j as usize].clone());
            }
            let new_leaf = Leaf { classes: new_leaf };
            self.leaves[j as usize] = new_leaf;
        }
    }

    pub fn train(&mut self, dataset: &EncryptedDataset, public_key: &PublicKey, ctx: &Context) {
        for sample in dataset.records.iter() {
            // Compile the tree
            let ctree = CTree::new(self, &sample.features, public_key, ctx);
            // Traverse the tree
            let selector = ctree.evaluate(public_key, ctx);
            // Update the leaves
            self.leaves_update(&selector, &sample.class, public_key, ctx);
        }
    }

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
        for leaf in self.leaves.iter() {
            print!("[");
            for (i, c) in leaf.classes.iter().enumerate() {
                if i == leaf.classes.len() - 1 {
                    print!("{}", c.to_byte(ctx, private_key));
                } else {
                    print!("{},", c.to_byte(ctx, private_key));
                }
            }
            print!("]");
        }
        println!("-----------------------------");
    }

    // pub fn to_json(&self, ctx: &Context)
    // pub fn save_to_file(&self, filepath: &str, ctx: &Context)

    // pub fn from_json(json: &serde_json::Value, ctx: &Context)
    // pub fn load_from_file(filepath: &str, ctx: &Context)
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
        let tree = Tree::new_random_tree(depth, n_classes, f, &public_key, &ctx);

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
