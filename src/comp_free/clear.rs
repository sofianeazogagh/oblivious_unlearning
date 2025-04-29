use crate::{comp_free::dataset::*, comp_free::DEBUG, VERBOSE};
use bincode::de;
use rand::{distributions::uniform::SampleBorrow, Rng, RngCore};
use revolut::{key, Context};
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
    pub final_leaves: Vec<u64>,
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
            final_leaves: Vec::new(),
        }
    }

    pub fn infer(&mut self, record: &ClearSample) -> &mut ClearLeaf {
        let mut current_node = &ClearInternalNode {
            id: 0,
            threshold: 0,
            feature_index: 0,
        };

        if self.root.threshold < record.features[self.root.feature_index as usize] {
            current_node = &self.nodes[0][0];
        } else {
            current_node = &self.nodes[0][1];
        }

        for idx in 1..self.depth - 1 {
            if current_node.threshold < record.features[current_node.feature_index as usize] {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize];
            } else {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize + 1];
            }
        }

        let mut selected_leaf: &mut ClearLeaf =
            if current_node.threshold < record.features[current_node.feature_index as usize] {
                &mut self.leaves[2 * current_node.id as usize]
            } else {
                &mut self.leaves[2 * current_node.id as usize + 1]
            };

        if DEBUG {
            println!("[CLEAR] Sample: {:?}", record.features);
            println!("[CLEAR] Selected leaf: {:?}", selected_leaf);
        }

        selected_leaf
    }

    pub fn train(&mut self, dataset: &ClearDataset, label_order: u64) {
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

            if label_order == 0 {
                // if the counts are all 0, set the label to the highest class number
                if leaf.counts.iter().sum::<u64>() == 0 {
                    leaf.label = dataset.n_classes;
                } else {
                    leaf.label = max_index as u64;
                }
            } else {
                leaf.label = max_index as u64;
            }
        }
    }

    pub fn update_statistic(&mut self, sample: &ClearSample) {
        let class_index = sample.class as usize;
        let selected_leaf = self.infer(&sample);
        selected_leaf.counts[class_index] += 1;
        println!("[CLEAR] Sample : {:?}", sample.features);
        self.print();
    }

    pub fn print(&self) {
        // println!();
        print!("---------- (f,t) ----------\n");
        self.root.print();
        for level in &self.nodes {
            for node in level {
                node.print();
            }
            println!();
        }
        // print!("\n---------([c0, c1, ..., cn], label)-----------\n");
        for class in 0..self.n_classes {
            for leaf in &self.leaves {
                // print!("[{:>2?}]", leaf.counts[class as usize]);
                print!("[{:02X}]", leaf.counts[class as usize]);
            }
            println!();
        }
        println!("\n");
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
            let leaf = ClearLeaf {
                counts,
                id: 0,
                label: 0,
            };
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
    pub fn new_random_forest(
        n_trees: u64,
        depth: u64,
        n_classes: u64,
        max_features: u64,
        f: u64,
    ) -> ClearForest {
        let mut forest = ClearForest { trees: Vec::new() };
        let mut rng = rand::thread_rng();
        for _ in 0..n_trees {
            let tree = ClearTree::generate_clear_random_tree(depth, n_classes, max_features, f);
            forest.trees.push(tree);
        }
        forest
    }

    pub fn train(&mut self, dataset: &ClearDataset, label_order: u64) {
        for tree in self.trees.iter_mut() {
            tree.train(dataset, label_order);
            tree.final_leaves = tree.leaves.iter().map(|leaf| leaf.label).collect();
        }
    }

    pub fn evaluate(&mut self, dataset: &ClearDataset) -> f64 {
        let mut correct = 0;
        let mut total = 0;
        let mut abstention = 0;

        for (i, sample) in dataset.records.iter().enumerate() {
            let mut counts = vec![0; self.trees[0].n_classes as usize + 1];
            for tree in self.trees.iter_mut() {
                let predicted_class = tree.infer(&sample);
                counts[predicted_class.label as usize] += 1;
            }

            // Argmax de counts
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
            if max_index as u64 == self.trees[0].n_classes {
                abstention += 1;
            }
            total += 1;
        }
        (correct as f64 + abstention as f64) / total as f64
    }

    pub fn fit_dataset(
        train_dataset: &ClearDataset,
        test_dataset: &ClearDataset,
        dataset_name: &str,
    ) -> (ClearForest, f64) {
        let num_trees = 64;
        let depth = 4;
        let max_features = train_dataset.max_features;
        let mut n_classes = 3;
        let mut f = 4;

        let num_trials = 100;
        let mut best_accuracy = 0.0;
        let mut best_model =
            ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);

        for _ in 0..num_trials {
            let mut forest =
                ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);
            forest.train(&train_dataset, 1);
            let accuracy = forest.evaluate(&test_dataset);
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_model = forest;
            }
        }
        println!("Best accuracy: {}", best_accuracy);

        (best_model, best_accuracy)
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_train_clear_forest() {
        let seed = 10;
        let filepath = "./src/comp_free/test_forest.json";
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = ClearDataset::from_file("data/iris-uci/iris.csv".to_string());
        let (train_dataset, test_dataset) = dataset.split(0.8);

        let mut forest = ClearForest::load_from_file(filepath, &ctx, &public_key);
        forest.train(&train_dataset, 1);
        let accuracy = forest.evaluate(&test_dataset);
        println!("Accuracy: {}", accuracy);
    }

    #[test]
    fn find_best_model() {
        // let dataset_name = "iris";
        let dataset_name = "wine";
        // let dataset_name = "adult";
        let seed = 10;
        let dataset_path = format!("data/{}-uci/{}.csv", dataset_name, dataset_name);
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = ClearDataset::from_file(dataset_path.to_string());
        let (train_dataset, test_dataset) = dataset.split(0.8);

        let num_trees = 64;
        let depth = 4;
        let max_features = dataset.max_features;
        let mut n_classes = 3;
        let mut f = 4;

        if dataset_name == "adult" {
            n_classes = 2;
            f = 105;
        }

        if dataset_name == "wine" {
            n_classes = 3;
            f = 13;
        }

        let num_trials = 100;
        let mut best_accuracy = 0.0;
        let mut best_model =
            ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);

        for _ in 0..num_trials {
            let mut forest =
                ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);
            forest.train(&train_dataset, 1);
            let accuracy = forest.evaluate(&test_dataset);
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_model = forest;
            }
        }

        let filepath = format!(
            "./src/comp_free/best_{}_{}_{}_{:.2}.json",
            dataset_name, num_trees, depth, best_accuracy
        );

        // best_model.save_to_file(&filepath);
        // println!("Best model saved to: {}", filepath);

        println!("Best accuracy: {}", best_accuracy);
    }

    #[test]
    fn test_argmax_order() {
        let dataset_name = "iris";
        let num_trees = 64;
        let depth = 4;
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = ClearDataset::from_file("data/iris-uci/iris.csv".to_string());
        let (train_dataset, test_dataset) = dataset.split(0.8);

        // find the best model
        let (mut forest, best_accuracy) =
            ClearForest::fit_dataset(&train_dataset, &test_dataset, dataset_name);

        // reset the leaves
        forest.trees.iter_mut().for_each(|tree| {
            tree.final_leaves = vec![0; tree.leaves.len()];
            tree.leaves.iter_mut().enumerate().for_each(|(i, leaf)| {
                leaf.counts = vec![0; tree.n_classes as usize];
            });
        });

        // train the model by choosing the highest label when all counts are 0
        forest.train(&train_dataset, 0);

        let accuracy = forest.evaluate(&test_dataset);
        println!("Best accuracy: {}", best_accuracy);
        println!("Accuracy with weird majority voting: {}", accuracy);
    }

    #[test]
    fn analysis_best_model() {
        // let dataset_name = "iris";
        // let dataset_name = "wine";
        let dataset_name = "cancer";
        // let dataset_name = "adult";
        let seed = 10;
        let dataset_path = format!("data/{}-uci/{}.csv", dataset_name, dataset_name);
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let dataset = ClearDataset::from_file(dataset_path.to_string());
        let (train_dataset, test_dataset) = dataset.split(0.8);

        let num_trees = 64;
        let depth = 4;
        let max_features = dataset.max_features;
        let mut n_classes = 3;
        let mut f = 4;

        if dataset_name == "adult" {
            n_classes = 2;
            f = 105;
        }

        if dataset_name == "wine" {
            n_classes = 3;
            f = 13;
        }

        if dataset_name == "cancer" {
            n_classes = 2;
            f = 30;
        }

        let num_trials = 100;
        let mut best_accuracy = 0.0;
        let mut best_model =
            ClearForest::new_random_forest(num_trees, depth, n_classes, max_features, f);
        let num_trees = [8, 16, 32, 64];
        for m in num_trees {
            for i in 0..num_trials {
                let mut forest =
                    ClearForest::new_random_forest(m, depth, n_classes, max_features, f);
                forest.train(&train_dataset, 1);
                let accuracy = forest.evaluate(&test_dataset);

                println!("Accuracy trial {}: {}", i, accuracy);

                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_model = forest;
                }
            }
            println!("Best accuracy for {} trees: {}", m, best_accuracy);

            let filepath = format!(
                "./src/comp_free/test_best_model/best_{}_{}_{}_{:.2}.json",
                dataset_name, m, depth, best_accuracy
            );
        }
    }
}
