// REVOLUT
use revolut::*;

use crate::ClearTree;

#[derive(Clone)]
pub struct Root {
    pub threshold: u64,
    pub feature_index: u64,
}

#[allow(dead_code)]
impl Root {
    pub fn print(&self) {
        println!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct InternalNode {
    pub threshold: u64,
    pub feature_index: u64,
}

#[allow(dead_code)]
impl InternalNode {
    pub fn print(&self) {
        print!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct Leaf {
    pub counts: LUT,
}

#[allow(dead_code)]
impl Leaf {
    pub fn print(&self, private_key: &PrivateKey, ctx: &Context, n_classes: u64) {
        // self.counts.print(private_key, ctx);
        let array = self.counts.to_array(private_key, ctx);
        print!("{:?}", &array[..n_classes as usize]);
    }
}

pub struct Tree {
    pub root: Root,
    pub nodes: Vec<Vec<InternalNode>>,
    pub leaves: Vec<Leaf>,
    pub depth: u64,
    pub n_classes: u64,
}

#[allow(dead_code)]
impl Tree {
    pub fn new() -> Self {
        Self {
            root: Root {
                threshold: 0,
                feature_index: 0,
            },
            nodes: Vec::new(),
            leaves: Vec::new(),
            depth: 0,
            n_classes: 0,
        }
    }

    pub fn generate_random_tree(depth: u64, n_classes: u64, f: u64, ctx: &Context) -> Self {
        let mut tree = Self::new();
        tree.depth = depth;
        tree.n_classes = n_classes;

        // Generate the root
        tree.root.threshold = rand::random::<u64>() % f as u64;
        tree.root.feature_index = rand::random::<u64>() % f;

        // Generate the nodes
        // Generate internal nodes for each level
        for _ in 1..depth {
            let mut stage = Vec::new();

            // Number of nodes at this level is 2^level_index
            let num_nodes = 2u64.pow(tree.nodes.len() as u32 + 1);

            for _ in 0..num_nodes {
                stage.push(InternalNode {
                    threshold: rand::random::<u64>() % f as u64,
                    feature_index: rand::random::<u64>() % f,
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

    pub fn print_tree(&self, private_key: &PrivateKey, ctx: &Context) {
        println!("-----------[(t,f)]-----------");
        self.root.print();
        for stage in self.nodes.iter() {
            for node in stage {
                node.print();
                print!(" ");
            }
            println!("");
        }
        for leaf in self.leaves.iter() {
            leaf.print(private_key, ctx, self.n_classes);
            print!(" ");
        }
        println!("");
        println!("-----------------------------");
    }

    pub fn sum_samples_luts_counts(
        &mut self,
        luts_samples: &Vec<Vec<LUT>>,
        public_key: &PublicKey,
    ) {
        let n_samples = luts_samples.len();
        let n_leaves = luts_samples[0].len();

        for i in 0..n_leaves {
            let mut sum_lut = luts_samples[0][i].clone();
            for j in 1..n_samples {
                public_key.glwe_sum_assign(&mut sum_lut.0, &luts_samples[j][i].0);
            }
            self.leaves[i].counts = sum_lut;
        }
    }

    #[allow(dead_code)]
    pub fn to_json(&self, ctx: &Context) -> serde_json::Value {
        let mut tree_json = serde_json::Map::new();

        // Serialize depth and n_classes
        tree_json.insert(
            "depth".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.depth)),
        );
        tree_json.insert(
            "n_classes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_classes)),
        );

        // Serialize leaves
        let private_key = key(ctx.parameters());
        let leaves_json: Vec<serde_json::Value> = self
            .leaves
            .iter()
            .map(|leaf| {
                serde_json::json!({
                    "type": "leaf",
                    "counts": leaf.counts.to_array(&private_key, ctx)[..self.n_classes as usize].to_vec()
                })
            })
            .collect();
        tree_json.insert("leaves".to_string(), serde_json::Value::Array(leaves_json));

        // Serialize internal nodes
        let mut nodes_json = Vec::new();
        for stage in &self.nodes {
            let stage_json: Vec<serde_json::Value> = stage
                .iter()
                .map(|node| {
                    serde_json::json!({
                        "threshold": node.threshold,
                        "feature_index": node.feature_index
                    })
                })
                .collect();
            nodes_json.push(stage_json);
        }
        tree_json.insert(
            "nodes".to_string(),
            serde_json::Value::Array(nodes_json.concat()),
        );

        // Serialize root
        let root_json = serde_json::json!({
            "threshold": self.root.threshold,
            "feature_index": self.root.feature_index
        });
        tree_json.insert("root".to_string(), root_json);

        serde_json::Value::Object(tree_json)
    }

    #[allow(dead_code)]
    pub fn save_to_file(&self, filepath: &str, ctx: &Context) {
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(filepath).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let json_value = self.to_json(ctx);
        let json_string = serde_json::to_string_pretty(&json_value).unwrap();
        std::fs::write(filepath, json_string).unwrap();
    }

    #[allow(dead_code)]
    pub fn from_json(json: &serde_json::Value, ctx: &Context) -> Self {
        let mut tree = Tree::new();

        // Deserialize depth and n_classes
        let depth = json.get("depth").and_then(|v| v.as_u64()).unwrap_or(0) as u64;
        let n_classes = json.get("n_classes").and_then(|v| v.as_u64()).unwrap_or(0) as u64;
        tree.depth = depth;
        tree.n_classes = n_classes;

        // Deserialize root
        if let Some(root) = json.get("root") {
            tree.root.threshold = root.get("threshold").and_then(|v| v.as_u64()).unwrap_or(0);
            tree.root.feature_index = root
                .get("feature_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }

        // Deserialize internal nodes
        if let Some(nodes) = json.get("nodes").and_then(|v| v.as_array()) {
            let mut node_index = 0;
            for level in 1..depth {
                let mut stage = Vec::new();
                let num_nodes = 2u64.pow(level as u32);

                for _ in 0..num_nodes {
                    if let Some(node_json) = nodes.get(node_index) {
                        let threshold = node_json
                            .get("threshold")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let feature_index = node_json
                            .get("feature_index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        stage.push(InternalNode {
                            threshold,
                            feature_index,
                        });
                        node_index += 1;
                    }
                }
                tree.nodes.push(stage);
            }
        }

        // Deserialize leaves
        if let Some(leaves) = json.get("leaves").and_then(|v| v.as_array()) {
            for leaf_json in leaves {
                let mut counts = vec![0; n_classes as usize];
                if let Some(counts_json) = leaf_json.get("counts").and_then(|v| v.as_array()) {
                    for (i, count) in counts_json.iter().enumerate() {
                        counts[i] = count.as_u64().unwrap_or(0);
                    }
                }
                tree.leaves.push(Leaf {
                    counts: LUT::from_vec_trivially(&counts, ctx),
                });
            }
        }

        tree
    }

    #[allow(dead_code)]
    pub fn load_from_file(filepath: &str, ctx: &Context) -> std::io::Result<Self> {
        let json_content = std::fs::read_to_string(filepath).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Tree file does not exist")
        })?;
        let json_value: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::from_json(&json_value, ctx))
    }
}

// Structure for the tree in the LUT format
pub struct TreeLUT {
    pub root: Root,
    pub stages: Vec<(LUT, LUT)>, // stages[i] contains (lut_index, lut_threshold) for the i-th stage
    pub leaves: Vec<LUT>,        // leaves[i] contains the lut for the i-th class
    pub n_classes: u64,
    pub depth: u64,
}

#[allow(dead_code)]
impl TreeLUT {
    pub fn new() -> Self {
        Self {
            root: Root {
                threshold: 0,
                feature_index: 0,
            },
            stages: Vec::new(),
            leaves: Vec::new(),
            n_classes: 0,
            depth: 0,
        }
    }

    pub fn from_tree(tree: &Tree, depth: u64, n_classes: u64, ctx: &Context) -> Self {
        let mut stages = Vec::new();

        // Pour chaque niveau de l'arbre (sauf la racine)
        for level in 0..depth - 1 {
            let mut thresholds = Vec::new();
            let mut feature_indices = Vec::new();

            for internal_node in tree.nodes[level as usize].iter() {
                thresholds.push(internal_node.threshold);
                feature_indices.push(internal_node.feature_index);
            }

            let lut_threshold = LUT::from_vec_trivially(&thresholds, ctx);
            let lut_index = LUT::from_vec_trivially(&feature_indices, ctx);
            stages.push((lut_index, lut_threshold));
        }

        // Initialiser les feuilles
        let leaves = (0..n_classes)
            .map(|_| LUT::from_vec_trivially(&vec![0], ctx))
            .collect();

        Self {
            root: tree.root.clone(),
            stages,
            leaves,
            n_classes,
            depth,
        }
    }

    pub fn from_clear_tree(tree: &ClearTree, ctx: &Context) -> Self {
        let mut stages = Vec::new();

        // Pour chaque niveau de l'arbre (sauf la racine)
        for level in 0..tree.depth - 1 {
            let mut thresholds = Vec::new();
            let mut feature_indices = Vec::new();

            for internal_node in tree.nodes[level as usize].iter() {
                thresholds.push(internal_node.threshold as u64);
                feature_indices.push(internal_node.feature_index);
            }

            let lut_threshold = LUT::from_vec_trivially(&thresholds, ctx);
            let lut_index = LUT::from_vec_trivially(&feature_indices, ctx);
            stages.push((lut_index, lut_threshold));
        }

        // Initialiser les feuilles
        let leaves = (0..tree.n_classes)
            .map(|_| LUT::from_vec_trivially(&vec![0], ctx))
            .collect();

        Self {
            root: Root {
                threshold: tree.root.threshold as u64,
                feature_index: tree.root.feature_index,
            },
            stages,
            leaves,
            n_classes: tree.n_classes,
            depth: tree.depth,
        }
    }

    pub fn generate_random_tree(depth: u64, n_classes: u64, dim: u64, ctx: &Context) -> Self {
        let mut tree = Self::new();
        tree.depth = depth;
        tree.n_classes = n_classes;

        // Generate the root
        tree.root.threshold = rand::random::<u64>() % ctx.full_message_modulus() as u64;
        tree.root.feature_index = rand::random::<u64>() % dim;

        // Generate the nodes
        // Generate internal nodes for each level
        for _ in 1..depth {
            let mut vec_index = Vec::new();
            let mut vec_threshold = Vec::new();

            // Number of nodes at this level is 2^level
            let num_nodes = 2u64.pow(tree.stages.len() as u32 + 1);

            for _ in 0..num_nodes {
                vec_index.push(rand::random::<u64>() % dim);
                vec_threshold.push(rand::random::<u64>() % ctx.full_message_modulus() as u64);
            }
            tree.stages.push((
                LUT::from_vec_trivially(&vec_index, ctx),
                LUT::from_vec_trivially(&vec_threshold, ctx),
            ));
        }

        // Generate the leaves
        let num_leaves = n_classes;
        for _ in 0..num_leaves {
            let counts = LUT::from_vec_trivially(&vec![0u64; n_classes as usize], ctx);
            tree.leaves.push(counts);
        }

        tree
    }

    pub fn print_tree(&self, private_key: &PrivateKey, ctx: &Context) {
        println!(
            "Root: ({}, {})",
            self.root.feature_index, self.root.threshold
        );
        for (i, stage) in self.stages.iter().enumerate() {
            let number_of_nodes = 2u64.pow(i as u32 + 1);

            let thresholds = stage.1.to_array(&private_key, ctx);
            let feature_indices = stage.0.to_array(&private_key, ctx);

            let thresholds = &thresholds[0..number_of_nodes as usize];
            let feature_indices = &feature_indices[0..number_of_nodes as usize];
            for j in 0..number_of_nodes {
                print!(
                    "({:?}, {:?})    ",
                    feature_indices[j as usize], thresholds[j as usize]
                );
            }
            println!();
        }

        println!("Leaves:");
        for (i, leaf) in self.leaves.iter().enumerate() {
            let class_counts = leaf.to_array(&private_key, ctx);
            println!("Leaf {}: {:?}", i, class_counts);
        }
    }
}
