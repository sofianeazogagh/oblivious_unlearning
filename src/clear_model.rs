use bincode::de;
use rand::{Rng, RngCore};

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
    pub label: u64,
}

impl Leaf {
    pub fn print(&self, n_classes: u64) {
        print!("({:?}, {})", &self.counts[..n_classes as usize], self.label);
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

    pub fn update_statistic(&mut self, sample: Vec<u64>){

        let mut current_node = &InternalNode {
            id: 0,
            threshold: 0,
            feature_index: 0,
        };

        if self.root.threshold <= sample[self.root.feature_index as usize] {
            current_node = &self.nodes[0][0];
        } else {
            current_node = &self.nodes[0][1];
        }

        // current_node.print();

        for idx in 1..self.depth - 1 {
            if current_node.threshold <= sample[current_node.feature_index as usize] {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize];
            } else {
                current_node = &self.nodes[idx as usize][2 * current_node.id as usize + 1];
            }
            // current_node.print();
        }

        let mut selected_leaf = & mut Leaf {
            counts: vec![0; self.n_classes as usize],
            label: 0,
        };

        if current_node.threshold <= sample[current_node.feature_index as usize] {
            selected_leaf = &mut self.leaves[2 * current_node.id as usize];
        } else {
            selected_leaf = &mut self.leaves[2 * current_node.id as usize + 1];
        }

        let class_index = sample[sample.len() - 1] as usize;

        selected_leaf.counts[class_index] += 1;

    }

    pub fn assign_label_to_leafs(&mut self) {
        for leaf in &mut self.leaves {
            let mut max = 0;
            let mut max_index = 0;
            for (i, count) in leaf.counts.iter().enumerate() {
                if *count > max {
                    max = *count;
                    max_index = i;
                }
            }
            leaf.label = max_index as u64;
        }
    }
    
    pub fn print_tree(&self) {
        print!("---------- (t,f) ----------\n");
        self.root.print();
        for level in &self.nodes {
            for node in level {
                node.print();
            }
            println!();
        }
        for leaf in &self.leaves {
            leaf.print(self.n_classes);
        }
        print!("\n")
    }
}

pub struct ClearDataset {
    pub records: Vec<Vec<u64>>,
    pub column_domains: Vec<(u64, u64)>,
    pub f: u64,
    pub n: u64,
}

impl ClearDataset {
    pub fn from_file(filepath: String) -> Self {
        let mut rdr = csv::Reader::from_path(filepath).unwrap();
        let mut records = Vec::new();
        let mut column_domains:Vec<(u64, u64)> = Vec::new();
        let mut n = 0;

        for result in rdr.records() {
            let record = result.unwrap();
            let mut record_vec = Vec::new();
            for (i, field) in record.iter().enumerate() {
                record_vec.push(field.parse::<u64>().unwrap());
            }
            records.push(record_vec);
            n += 1;
        }

        let mut column_domains = Vec::new();
        for i in 0..records[0].len() {
            let mut min = std::u64::MAX;
            let mut max = std::u64::MIN;
            for record in &records {
                if record[i] < min {
                    min = record[i];
                }
                if record[i] > max {
                    max = record[i];
                }
            }

            if max == 0 {
                max = 1
            }
            column_domains.push((min, max));
        }

        let f = column_domains.len() as u64;

        Self {
            records,
            column_domains,
            f,
            n,
        }
    }
}

pub fn generate_clear_random_tree(depth: u64, n_classes: u64, column_domains: Vec<(u64, u64)>, f: u64) -> ClearTree {
    let mut tree = ClearTree::new();
    tree.depth = depth;
    tree.n_classes = n_classes;

    let mut rng = rand::thread_rng();

    // the feature index should be selected among the possible feature indices
    tree.root.feature_index =  rng.gen_range(0..f);
    let mut feature_domain = column_domains[tree.root.feature_index as usize];
    // print!("{:?}", feature_domain);

    tree.root.threshold = rng.gen_range(feature_domain.0..feature_domain.1);

    for idx in 1..depth {
        let mut level = Vec::new();
        for j in 0..(2u64.pow(idx as u32) as usize) {
            let feature_index = rng.gen_range(0..f);
            feature_domain = column_domains[feature_index as usize];
            // print!("{:?}", feature_domain);
            let node = InternalNode {
                id: j as u64,
                threshold: rng.gen_range(feature_domain.0..feature_domain.1),
                feature_index,
            };
            level.push(node);
        }
        tree.nodes.push(level);
    }

    for _ in 0..(2u64.pow(depth as u32) as usize) {
        let counts = vec![0; n_classes as usize];
        let leaf = Leaf { counts, label: 0 };
        tree.leaves.push(leaf);
    }

    tree
}






