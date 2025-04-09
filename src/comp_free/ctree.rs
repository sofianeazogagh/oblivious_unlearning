use crate::*;

use super::dataset::*;
use super::tree::*;

pub struct CTree {
    root: LWE,
    stages: Vec<Vec<LWE>>,
}

impl CTree {
    pub fn new(
        tree: &Tree,
        sample: &EncryptedSample,
        public_key: &PublicKey,
        ctx: &Context,
        private_key: &PrivateKey,
    ) -> Self {
        let features = sample.features.clone();
        let classes = sample.class.clone();

        // 1. for each node in the tree, get the comp results from the sample
        // - Root
        let feature_index = tree.root.index;
        let threshold = tree.root.threshold;
        let b = public_key.glwe_extract(&features[feature_index as usize], threshold as usize, ctx);
        let mut ctree = Self {
            root: b,
            stages: Vec::new(),
        };

        private_key.debug_lwe("Root", &ctree.root, ctx);

        // - Stages
        for stage in tree.stages.iter() {
            let mut stage_results = Vec::new();
            for node in stage.iter() {
                let feature_index = node.index;
                let threshold = node.threshold;
                let b = public_key.glwe_extract(
                    &features[feature_index as usize],
                    threshold as usize,
                    ctx,
                );
                stage_results.push(b);
            }
            ctree.stages.push(stage_results);
        }

        // 2. return the ctree
        ctree
    }

    /// Evaluate the CTree on a given sample.
    /// Returns the selector of the leaf (to be used either for inference or training)
    pub fn evaluate(&self, public_key: &PublicKey, ctx: &Context) -> LWE {
        // 1. Initialize the selector with the root
        let mut selector = self.root.clone();

        // 2. For each stage
        for stage in self.stages.iter() {
            // Pack the stage into a LUT
            let lut_stages = LUT::from_vec_of_lwe(stage, public_key, ctx);

            // Blindly select the correct node
            let b = public_key.blind_array_access(&selector, &lut_stages, ctx);

            // Update the selector with the formula s_i = b + 2 * s_{i-1}
            selector = public_key.lwe_mul_add(&b, &selector, 2);
        }

        // 3. Return the final selector
        selector
    }

    pub fn print(&self, private_key: &PrivateKey, ctx: &Context) {
        println!("-----------[ LWE(b) ]-----------");
        let root = private_key.decrypt_lwe(&self.root, ctx);
        println!("Root: {}", root);
        for stage in self.stages.iter() {
            for node in stage.iter() {
                let b = private_key.decrypt_lwe(node, ctx);
                print!("{} ", b);
            }
            println!("");
        }
        println!("-----------------------------");
    }
}
