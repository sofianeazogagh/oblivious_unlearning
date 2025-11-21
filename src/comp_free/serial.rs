// src/comp_free/serial.rs

use crate::comp_free::clear::*;
use crate::comp_free::forest::Forest;
use crate::comp_free::tree::{Classes, Node, Tree};
use crate::ByteLWE;
use crate::LWE;
use crate::PARAM_MESSAGE_4_CARRY_0;
use revolut::radix::NyblByteLUT;
use revolut::{Context, PrivateKey, PublicKey};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

impl Tree {
    pub fn to_json(&self, private_key: &PrivateKey, ctx: &Context) -> Value {
        let root = json!({
            "threshold": self.root.threshold,
            "index": self.root.index,
        });

        let stages: Vec<Value> = self
            .stages
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|node| {
                        json!({
                            "threshold": node.threshold,
                            "index": node.index,
                        })
                    })
                    .collect()
            })
            .collect();

        // Unpack de LUTs into the Vec of Vec<ByteLWE> : one per class
        let mut classes = Vec::new();
        for i in 0..self.n_classes {
            classes.push(
                self.leaves_lut[i as usize]
                    .class
                    .to_many_blwes(&private_key.public_key, ctx),
            );
        }
        // Push the classes into the leaves : n_classes per leaf
        let mut leaves: Vec<Vec<ByteLWE>> = Vec::new();
        for j in 0..2u64.pow(self.depth as u32) as usize {
            let mut leaf = Vec::new();
            for i in 0..self.n_classes {
                leaf.push(classes[i as usize][j as usize].clone());
            }
            leaves.push(leaf);
        }

        let leaves: Vec<Value> = leaves
            .iter()
            .map(|leaf| {
                let classes: Vec<u8> = leaf.iter().map(|c| c.to_byte(ctx, private_key)).collect();
                json!({ "classes": classes })
            })
            .collect();

        let final_leaves: Vec<u64> = self
            .final_leaves
            .iter()
            .map(|c| private_key.decrypt_lwe(c, ctx))
            .collect();

        json!({
            "root": root,
            "stages": stages,
            "leaves": leaves,
            "final_leaves": final_leaves,
            "depth": self.depth,
            "n_classes": self.n_classes,
        })
    }

    pub fn from_json(json: &Value, ctx: &Context, public_key: &PublicKey) -> Self {
        let n_classes = json["n_classes"].as_u64().unwrap();
        let depth = json["depth"].as_u64().unwrap();
        let root = Node {
            threshold: json["root"]["threshold"].as_u64().unwrap(),
            index: json["root"]["index"].as_u64().unwrap(),
        };

        let stages: Vec<Vec<Node>> = json["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|stage| {
                stage
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|node| Node {
                        threshold: node["threshold"].as_u64().unwrap(),
                        index: node["index"].as_u64().unwrap(),
                    })
                    .collect()
            })
            .collect();

        let leaves: Vec<Vec<u8>> = json["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|leaf| {
                let classes: Vec<u8> = leaf["classes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        // ByteLWE::from_byte_trivially(c.as_u64().unwrap() as u8, ctx, public_key)
                        c.as_u64().unwrap() as u8
                    })
                    .collect();
                classes
            })
            .collect();

        // Pack the classes into the leaves : one per class
        let mut leaves_lut = Vec::new();
        for i in 0..n_classes {
            let mut class = Vec::new();
            for j in 0..2u64.pow(depth as u32) as usize {
                class.push(leaves[j as usize][i as usize]);
            }
            let mut class_slice = [0u8; 16];
            for (k, &value) in class.iter().enumerate() {
                class_slice[k] = value;
            }
            leaves_lut.push(Classes {
                class: NyblByteLUT::from_bytes_trivially(&class_slice, ctx),
            });
        }

        let final_leaves: Vec<LWE> = json["final_leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| public_key.allocate_and_trivially_encrypt_lwe(c.as_u64().unwrap(), ctx))
            .collect();

        Tree {
            root,
            stages,
            leaves_lut,
            depth,
            n_classes,
            final_leaves,
        }
    }


    pub fn new_from_value(json: &serde_json::Value, public_key: &PublicKey, ctx: &Context) -> Self {
        let mut tree = Self::new(
            json.get("depth").and_then(|d| d.as_u64()).unwrap_or(0),
            json.get("n_classes").and_then(|c| c.as_u64()).unwrap_or(0),
        );

        tree.root.threshold = json
            .get("root")
            .and_then(|r| r.get("threshold").and_then(|t| t.as_u64()))
            .unwrap_or(0);
        tree.root.index = json
            .get("root")
            .and_then(|r| r.get("index").and_then(|i| i.as_u64()))
            .unwrap_or(0);

        for stage in json.get("stages").and_then(|s| s.as_array()) {
            for node in stage {
                tree.stages.push(
                    node.as_array()
                        .unwrap()
                        .iter()
                        .map(|n| Node {
                            threshold: n.get("threshold").and_then(|t| t.as_u64()).unwrap_or(0),
                            index: n.get("index").and_then(|i| i.as_u64()).unwrap_or(0),
                        })
                        .collect(),
                );
            }
        }

        for _ in 0..tree.n_classes {
            let class = NyblByteLUT::from_bytes_trivially(&[0x00; 16], ctx);
            tree.leaves_lut.push(Classes { class });
        }

        tree
    }

}

impl Forest {
    pub fn to_json(&self, private_key: &PrivateKey, ctx: &Context) -> Value {
        let trees: Vec<Value> = self
            .trees
            .iter()
            .map(|tree| tree.to_json(private_key, ctx))
            .collect();

        json!({ "trees": trees })
    }

    pub fn save_to_file(&self, filepath: &str, private_key: &PrivateKey, ctx: &Context) {
        if !Path::new(filepath).exists() {
            File::create(filepath).expect("Unable to create file");
        }
        let json_data = self.to_json(private_key, ctx);
        let json_string = serde_json::to_string_pretty(&json_data).unwrap();
        let mut file = File::create(filepath).expect("Unable to create file");
        file.write_all(json_string.as_bytes())
            .expect("Unable to write data");
    }

    pub fn load_from_file(filepath: &str, ctx: &Context, public_key: &PublicKey) -> Self {
        let mut file = File::open(filepath).expect("Unable to open file");
        let mut json_string = String::new();
        file.read_to_string(&mut json_string)
            .expect("Unable to read data");

        let json_data: Value = serde_json::from_str(&json_string).unwrap();
        let trees: Vec<Tree> = json_data["trees"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tree_json| Tree::from_json(tree_json, ctx, public_key))
            .collect();

        Forest { trees }
    }

    pub fn new_from_file(filepath: &str, public_key: &PublicKey, ctx: &Context) -> Self {
        let file = std::fs::File::open(filepath).expect("Unable to open file");
        let reader = std::io::BufReader::new(file);
        let json: serde_json::Value =
            serde_json::from_reader(reader).expect("Unable to parse JSON");

        let mut trees = Vec::new();
        for tree_json in json.get("trees").and_then(|t| t.as_array()).unwrap() {
            let tree = Tree::new_from_value(tree_json, public_key, ctx);
            trees.push(tree);
        }

        Self { trees }
    }


    pub fn save_perf_to_file(
        &self,
        file_path: &str,
        duration_train: Duration,
        duration_test: Duration,
        dataset_name: &str,
        n_trees: u64,
        depth: u64,
        accuracy: f64,
    ) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(file_path)
            .unwrap();

        writeln!(
            file,
            "{},{},{},{:?},{:?},{}",
            n_trees, depth, dataset_name, duration_train, duration_test, accuracy
        )
        .unwrap();
    }


    pub fn save_perf_for_bench(
        &self,
        file_path: &str,
        duration_train: Duration,
        duration_test: Duration,
        dataset_name: &str,
        n_trees: u64,
        depth: u64,
        accuracy: f64,
        real_accuracy: f64,
        best_accuracy: f64,
    ) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(file_path)
            .unwrap();

        writeln!(
            file,
            "{},{},{},{:?},{:?},{},{},{}",
            n_trees, depth, dataset_name, duration_train, duration_test, accuracy, real_accuracy, best_accuracy
        )
        .unwrap();
    }
}

impl ClearTree {
    pub fn from_json(json_data: &Value) -> Self {
        let root = ClearRoot {
            threshold: json_data["root"]["threshold"].as_u64().unwrap(),
            feature_index: json_data["root"]["index"].as_u64().unwrap(),
        };

        let nodes = json_data["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|stage| {
                stage
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, node)| ClearInternalNode {
                        threshold: node["threshold"].as_u64().unwrap(),
                        feature_index: node["index"].as_u64().unwrap(),
                        id: i as u64,
                    })
                    .collect()
            })
            .collect();

        let leaves = json_data["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, leaf)| ClearLeaf {
                counts: leaf["classes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|count| 
                        // count.as_u64().unwrap()
                        0
                    )
                    .collect(),
                id: i as u64,
                label: 0,
            })
            .collect();

        let final_leaves: Vec<u64> = json_data["final_leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();

        ClearTree {
            root,
            nodes,
            leaves,
            depth: json_data["depth"].as_u64().unwrap(),
            n_classes: json_data["n_classes"].as_u64().unwrap(),
            final_leaves,
        }
    }

    pub fn to_json(&self) -> Value {
        let root = json!({
            "threshold": self.root.threshold,
            "index": self.root.feature_index,
        });

        let stages: Vec<Value> = self.nodes.iter().map(|stage| {
            stage.iter().map(|node| {
                json!({
                    "threshold": node.threshold,
                    "index": node.feature_index,
                })
            })
            .collect()
        }).collect();

        let leaves: Vec<Value> = self.leaves.iter().map(|leaf| {
            let classes: Vec<u64> = leaf.counts.clone();
            json!({ "classes": classes })
        }).collect();

            
        json!({ "root": root, 
                "stages": stages, 
                "leaves": leaves,
                "depth": self.depth,
                "n_classes": self.n_classes,
                "final_leaves": self.final_leaves,
        })
    }
}

impl ClearForest {
    pub fn load_from_file(filepath: &str) -> Self {
        let mut file = File::open(filepath).expect("Unable to open file");
        let mut json_string = String::new();
        file.read_to_string(&mut json_string)
            .expect("Unable to read data");

        let json_data: Value = serde_json::from_str(&json_string).unwrap();
        let trees: Vec<ClearTree> = json_data["trees"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tree_json| ClearTree::from_json(tree_json))
            .collect();

        ClearForest { trees }
    }

    pub fn to_json(&self) -> Value {
        let trees: Vec<Value> = self.trees.iter().map(|tree| tree.to_json()).collect();
        json!({ "trees": trees })
    }

    pub fn save_to_file(&self, filepath: &str) {
        let json_data = self.to_json();
        let json_string = serde_json::to_string_pretty(&json_data).unwrap();
        let mut file = File::create(filepath).expect("Unable to create file");
        file.write_all(json_string.as_bytes())
            .expect("Unable to write data");
    }
}

#[cfg(test)]
mod tests {
    use crate::comp_free::clear;

    use super::*;
    use revolut::{key, Context, PrivateKey, PublicKey};
    use std::fs;

    #[test]
    fn test_forest_serialization_and_deserialization() {
        // Create a context
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Generate public and private keys
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        // Define parameters for the forest
        let n_trees = 64;
        let depth = 4;
        let n_classes = 3;
        let f = 4; // 4 features for the iris dataset

        // Create a new forest
        let forest = Forest::new(n_trees, depth, n_classes, f, &public_key, &ctx);

        // Define a file path for saving the forest
        let filepath = "./src/comp_free/test_forest_serial.json";

        // Save the forest to a file
        println!("Saving forest to file");
        forest.save_to_file(filepath, &private_key, &ctx);

        // Load the forest from the file
        println!("Loading forest from file");
        let loaded_forest = Forest::load_from_file(filepath, &ctx, &public_key);

        // // Clean up the test file
        // fs::remove_file(filepath).expect("Unable to delete test file");

        // Compare the original and loaded forests
        assert_eq!(forest.trees.len(), loaded_forest.trees.len());

        for (original_tree, loaded_tree) in forest.trees.iter().zip(loaded_forest.trees.iter()) {
            assert_eq!(original_tree.depth, loaded_tree.depth);
            assert_eq!(original_tree.n_classes, loaded_tree.n_classes);
            assert_eq!(original_tree.root.threshold, loaded_tree.root.threshold);
            assert_eq!(original_tree.root.index, loaded_tree.root.index);

            for (original_stage, loaded_stage) in
                original_tree.stages.iter().zip(loaded_tree.stages.iter())
            {
                for (original_node, loaded_node) in original_stage.iter().zip(loaded_stage.iter()) {
                    assert_eq!(original_node.threshold, loaded_node.threshold);
                    assert_eq!(original_node.index, loaded_node.index);
                }
            }
            for (original_leaf, loaded_leaf) in original_tree
                .leaves_lut
                .iter()
                .zip(loaded_tree.leaves_lut.iter())
            {
                assert_eq!(
                    original_leaf
                        .class
                        .to_bytes(&public_key, &private_key, &ctx),
                    loaded_leaf.class.to_bytes(&public_key, &private_key, &ctx)
                );
            }
        }
    }

    #[test]
    fn test_clear_forest_from_file() {
        let filepath = "./src/comp_free/test_forest.json";
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;
        let clear_forest = ClearForest::load_from_file(filepath);

        let forest = Forest::load_from_file(filepath, &ctx, public_key);

        assert_eq!(clear_forest.trees.len(), forest.trees.len());
        assert_eq!(clear_forest.trees[0].depth, forest.trees[0].depth);
        assert_eq!(clear_forest.trees[0].n_classes, forest.trees[0].n_classes);

        
        for i in 0.. clear_forest.trees.len() {
            for j in 0.. clear_forest.trees[i].nodes.len() {
                for k in 0.. clear_forest.trees[i].nodes[j].len() {
                    assert_eq!(clear_forest.trees[i].nodes[j][k].threshold, forest.trees[i].stages[j][k].threshold);
                    assert_eq!(clear_forest.trees[i].nodes[j][k].feature_index, forest.trees[i].stages[j][k].index);
                }

                
            }

            for k in 0.. clear_forest.trees[i].leaves.len() {
                for l in 0.. clear_forest.trees[i].leaves[k].counts.len() {
                    assert_eq!(clear_forest.trees[i].leaves[k].counts[l], forest.trees[i].leaves_lut[l].class
                        .at(k as u8, &ctx, &public_key)
                        .to_byte(&ctx, &private_key) as u64);
                }
            }
        }
        
        
    }



    #[test]
    fn test_new_forest_from_file() {
        let filepath = "./src/comp_free/best_iris.csv_64_4.json";
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;

        let forest = Forest::new_from_file(filepath, public_key, &ctx);

        assert_eq!(forest.trees.len(), 64); 

        for tree in forest.trees.iter() {
            assert_eq!(tree.depth, 4); 
            assert_eq!(tree.n_classes, 3); 

            for stage in tree.stages.iter() {
                for node in stage.iter() {
                    println!("node.threshold: {:?}", node.threshold);
                    assert!(node.threshold >= 0);
                    println!("node.index: {:?}", node.index);
                    assert!(node.index >= 0);
                }
            }

            for class in tree.leaves_lut.iter() {
                let bytes = class.class.to_bytes(&public_key, &private_key, &ctx);
                assert_eq!(bytes.len(), 16); 
            }
        }
    }
}
