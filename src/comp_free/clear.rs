use bincode::de;
use rand::{distributions::uniform::SampleBorrow, Rng, RngCore};
use crate::comp_free::dataset::*;
use revolut::{Context, key};
use tfhe::shortint::parameters::PARAM_MESSAGE_4_CARRY_0;

#[derive(Clone)]
pub struct ClearRoot {
    pub threshold: u64,
    pub feature_index: u64,
}

impl ClearRoot {
    pub fn print(&self) {
        println!("({},{})", self.feature_index, self.threshold,);
    }
}

#[derive(Clone)]
pub struct ClearInternalNode {
    pub id: u64,
    pub threshold: u64,
    pub feature_index: u64,
}

impl ClearInternalNode {
    pub fn print(&self) {
        print!("({},{})", self.feature_index, self.threshold);
    }
}

#[derive(Clone, Debug)]
pub struct ClearLeaf {
    pub counts: Vec<u64>,
    pub id: u64,
    pub label: u64,
}

impl ClearLeaf {
    pub fn print(&self, n_classes: u64) {
        print!("({:?}, {})", &self.counts[..n_classes as usize], self.label);
    }
}

#[derive(Clone)]
pub struct ClearTree {
    pub root: ClearRoot,
    pub nodes: Vec<Vec<ClearInternalNode>>,
    pub leaves: Vec<ClearLeaf>,
    pub depth: u64,
    pub n_classes: u64,
}

impl ClearTree {
    pub fn new() -> Self {
        Self {
            root: ClearRoot {
                threshold: 0,
                feature_index: 0,
            },
            nodes: Vec::new(),
            leaves: Vec::new(),
            depth: 0,
            n_classes: 0,
        }
    }

    pub fn infer(&mut self, record: &ClearSample) -> &mut ClearLeaf {
        let mut current_node = &ClearInternalNode {
            id: 0,
            threshold: 0,
            feature_index: 0,
        };

        if self.root.threshold <= record.features[self.root.feature_index as usize] {
            current_node = &self.nodes[0][0];
        } else {
            current_node = &self.nodes[0][1];
        }

        for idx in 1..self.depth - 1 {
            if current_node.threshold <= record.features[current_node.feature_index as usize] {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize];
            } else {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize + 1];
            }
        }

        let mut selected_leaf: &mut ClearLeaf =
            if current_node.threshold <= record.features[current_node.feature_index as usize] {
                &mut self.leaves[2 * current_node.id as usize]
            } else {
                &mut self.leaves[2 * current_node.id as usize + 1]
            };

        selected_leaf
    }


    pub fn train(&mut self, dataset: &ClearDataset) {
        for sample in dataset.records.iter() {
            self.update_statistic(sample);
        }


        // majoority voting to decide the class of each leaf
        for leaf in self.leaves.iter_mut() {
            leaf.counts = leaf.counts.iter().map(|count| *count as u64).collect();

            let mut max_count = 0;
            let mut max_index = 0;
            for (i, count) in leaf.counts.iter().enumerate() {
                if count > &max_count {
                    max_count = *count;
                    max_index = i;
                }
            }
            leaf.label = max_index as u64;
        }
    }

    pub fn update_statistic(&mut self, sample: &ClearSample) {
        let class_index = sample.class as usize;
        let selected_leaf = self.infer(&sample);
        selected_leaf.counts[class_index] += 1;
    }

    pub fn print(&self) {
        print!("---------- (f,t) ----------\n");
        self.root.print();
        for level in &self.nodes {
            for node in level {
                node.print();
            }
            println!();
        }
        print!("\n---------([c0, c1, ..., cn], label)-----------\n");
        for leaf in &self.leaves {
            leaf.print(self.n_classes);
        }

    }

    pub fn generate_clear_random_tree(
        depth: u64,
        n_classes: u64,
        max_features: u64,
        f: u64,
    ) -> ClearTree {
        let mut tree = ClearTree::new();
        tree.depth = depth;
        tree.n_classes = n_classes;

        let mut rng = rand::thread_rng();

        // the feature index should be selected among the possible feature indices
        // tree.root.feature_index = rng.gen_range(0..f);
        tree.root.feature_index = rand::random::<u64>() % f as u64;
        // tree.root.threshold = rng.gen_range(features_domain.0..=features_domain.1) as f64;
        tree.root.threshold = rand::random::<u64>() % max_features as u64;

        tree.root.print();

        for idx in 1..depth {
            let mut level = Vec::new();
            for j in 0..(2u64.pow(idx as u32) as usize) {
                let feature_index = rng.gen_range(0..f);
                let mut threshold = 0.0;
                let threshold = rand::random::<u64>() % max_features as u64;

                let node = ClearInternalNode {
                    id: j as u64,
                    threshold,
                    feature_index,
                };
                level.push(node);
            }
            tree.nodes.push(level);
        }

        for i in 0..(2u64.pow(depth as u32) as usize) {
            let counts = vec![0; n_classes as usize];
            let leaf = ClearLeaf { counts, id: 0, label: 0 };
            tree.leaves.push(leaf);
        }

        tree
    }
}

#[derive(Clone)]
pub struct ClearForest {
    pub trees: Vec<ClearTree>,
}

impl ClearForest {

    pub fn train(&mut self, dataset: &ClearDataset) {
        for tree in self.trees.iter_mut() {
            tree.train(dataset);
        }

        // for tree in self.trees.iter() {
        //     tree.print();
        // }
    }

    pub fn evaluate(&mut self, dataset: &ClearDataset) -> f64 {
        let mut correct = 0;
        let mut total = 0;
        
        for sample in dataset.records.iter() {
            let mut counts = vec![0; self.trees[0].n_classes as usize];
            for tree in self.trees.iter_mut() {
                let predicted_class = tree.infer(&sample);
                counts[predicted_class.label as usize] += 1;
            }

            let mut max_count = 0;
            let mut max_index = 0;
            for (i, count) in counts.iter().enumerate() {
                if count > &max_count {
                    max_count = *count;
                    max_index = i;
                }
            }
            if max_index as u64 == sample.class {
                correct += 1;
            }
            total += 1;
        }
        correct as f64 / total as f64
    }



}
    
    
mod tests {
    use super::*;

    #[test]
    fn test_train_clear_forest() {
        let filepath = "./src/comp_free/test_forest.json";
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = ClearDataset::from_file("data/iris-uci/iris.csv".to_string());
        let (train_dataset, test_dataset) = dataset.split(0.8);

        let mut forest = ClearForest::load_from_file(filepath, &ctx, &public_key);
        forest.train(&train_dataset);
        let accuracy = forest.evaluate(&test_dataset);
        println!("Accuracy: {}", accuracy);
        
    

    }
}
