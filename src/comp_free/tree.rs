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

pub struct Tree {
    pub root: Node,
    pub stages: Vec<Vec<Node>>, // Stages[i] is the i-th stages of the tree
    pub leaves: Vec<Vec<ByteLWE>>,
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
            let leaf =
                vec![ByteLWE::from_byte_trivially(0x00, ctx, public_key); n_classes as usize];
            tree.leaves.push(leaf);
        }

        tree
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
            for (i, c) in leaf.iter().enumerate() {
                if i == leaf.len() - 1 {
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
        let ctree = CTree::new(&tree, &encrypted_sample, &public_key, &ctx, &private_key);
        println!("CTree: ");
        ctree.print(&private_key, &ctx);

        // Evaluate the tree
        let selector = ctree.evaluate(&public_key, &ctx);
        println!("Selector: {}", private_key.decrypt_lwe(&selector, &ctx));
    }
}
