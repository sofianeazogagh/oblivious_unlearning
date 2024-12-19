use bincode::de;
use rand::{distributions::uniform::SampleBorrow, Rng, RngCore};

pub struct Root {
    pub threshold: u64,
    pub feature_index: u64,
}

impl Root {
    pub fn print(&self) {
        println!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct InternalNode {
    pub id: u64,
    pub threshold: u64,
    pub feature_index: u64,
}

impl InternalNode {
    pub fn print(&self) {
        print!("({},{})", self.threshold, self.feature_index);
    }
}

pub struct Leaf {
    pub counts: Vec<u64>,
    pub id: u64,
}

impl Leaf {
    pub fn print(&self, n_classes: u64) {
        print!("({:?}, {})", &self.counts[..n_classes as usize], self.id);
    }
}

pub struct ClearTree {
    pub root: Root,
    pub nodes: Vec<Vec<InternalNode>>,
    pub leaves: Vec<Leaf>,
    pub depth: u64,
    pub n_classes: u64,
}

impl ClearTree {
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

    pub fn infer(&mut self, record: Vec<u64>) -> &mut Leaf {
        let mut current_node = &InternalNode {
            id: 0,
            threshold: 0,
            feature_index: 0,
        };

        if self.root.threshold <= record[self.root.feature_index as usize] {
            current_node = &self.nodes[0][0];
        } else {
            current_node = &self.nodes[0][1];
        }

        for idx in 1..self.depth - 1 {
            if current_node.threshold <= record[current_node.feature_index as usize] {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize];
            } else {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize + 1];
            }
        }

        let mut selected_leaf: &mut Leaf =
            if current_node.threshold <= record[current_node.feature_index as usize] {
                &mut self.leaves[2 * current_node.id as usize]
            } else {
                &mut self.leaves[2 * current_node.id as usize + 1]
            };

        selected_leaf
    }

    pub fn update_statistic(&mut self, sample: Vec<u64>) {
        let class_index = sample[sample.len() - 1] as usize;
        let selected_leaf = self.infer(sample);
        selected_leaf.counts[class_index] += 1;
    }

    pub fn print(&self) {
        print!("---------- (t,f) ----------\n");
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
}

#[derive(Clone)]
pub struct ClearSample {
    pub features: Vec<u64>,
    pub class: u64,
}

impl ClearSample {
    pub fn print(&self) {
        print!("\n([");
        for feature in &self.features {
            print!("{},", feature);
        }
        print!("],{})", self.class);
    }
}

#[derive(Clone)]
pub struct ClearDataset {
    pub records: Vec<ClearSample>,
    pub features_domain: (u64, u64),
    pub n_classes: u64,
    pub f: u64,
    pub n: u64,
}

impl ClearDataset {
    pub fn from_file(filepath: String) -> Self {
        let mut rdr = csv::Reader::from_path(filepath).unwrap();
        let mut records = Vec::new();
        let mut n = 0;

        let mut min = std::u64::MAX;
        let mut max = std::u64::MIN;
        let mut classes = Vec::new();

        for result in rdr.records() {
            let record = result.unwrap();
            let mut record_vec = Vec::new();
            let mut class = 0;

            for (i, field) in record.iter().enumerate() {
                let value = field.parse::<u64>().unwrap();

                if i == record.len() - 1 {
                    class = value;
                    if !classes.contains(&class) {
                        classes.push(class);
                    }
                } else {
                    record_vec.push(value);
                }

                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }
            records.push(ClearSample {
                features: record_vec,
                class,
            });
        }

        let f = records[0].features.len() as u64;
        let features_domain = (min, max);

        Self {
            records,
            features_domain,
            n_classes: classes.len() as u64,
            f,
            n,
        }
    }

    pub fn split(&self, train: f64) -> (ClearDataset, ClearDataset) {
        let mut rng = rand::thread_rng();
        let n = self.records.len();
        let n_train = (train * n as f64) as u64;
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        for _ in 0..n_train {
            let idx = rng.gen_range(0..n);
            train_indices.push(idx);
        }
        for i in 0..n {
            if !train_indices.contains(&i) {
                test_indices.push(i);
            }
        }
        let mut train_records = Vec::new();
        let mut test_records = Vec::new();
        for idx in train_indices {
            train_records.push(self.records[idx].clone());
        }
        for idx in test_indices {
            test_records.push(self.records[idx].clone());
        }

        let train_dataset = ClearDataset {
            records: train_records,
            features_domain: self.features_domain,
            n_classes: self.n_classes,
            f: self.f,
            n: n_train,
        };

        let test_dataset = ClearDataset {
            records: test_records,
            features_domain: self.features_domain,
            n_classes: self.n_classes,
            f: self.f,
            n: n as u64 - n_train,
        };

        (train_dataset, test_dataset)
    }
}

pub fn generate_clear_random_tree(
    depth: u64,
    n_classes: u64,
    features_domain: (u64, u64),
    f: u64,
) -> ClearTree {
    let mut tree = ClearTree::new();
    tree.depth = depth;
    tree.n_classes = n_classes;

    let mut rng = rand::thread_rng();

    // the feature index should be selected among the possible feature indices
    tree.root.feature_index = rng.gen_range(0..f);

    tree.root.threshold = rng.gen_range(features_domain.0..=features_domain.1);

    for idx in 1..depth {
        let mut level = Vec::new();
        for j in 0..(2u64.pow(idx as u32) as usize) {
            let feature_index = rng.gen_range(0..f);
            let mut threshold = 0;
            if features_domain.0 == features_domain.1 {
                threshold = features_domain.0;
            } else {
                threshold = rng.gen_range(features_domain.0..=features_domain.1);
            }

            let node = InternalNode {
                id: j as u64,
                threshold: threshold,
                feature_index,
            };
            level.push(node);
        }
        tree.nodes.push(level);
    }

    for i in 0..(2u64.pow(depth as u32) as usize) {
        let counts = vec![0; n_classes as usize];
        let leaf = Leaf { counts, id: 0 };
        tree.leaves.push(leaf);
    }

    tree
}
