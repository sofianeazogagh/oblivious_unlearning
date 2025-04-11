// src/comp_free/serial.rs

use crate::comp_free::clear::*;
use crate::comp_free::forest::Forest;
use crate::comp_free::tree::{Leaf, Node, Tree};
use crate::radix::ByteLWE;
use revolut::{Context, PrivateKey, PublicKey};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Duration;
use crate::LWE;
use crate::PARAM_MESSAGE_4_CARRY_0;

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

        let leaves: Vec<Value> = self
            .leaves
            .iter()
            .map(|leaf| {
                let classes: Vec<u8> = leaf
                    .classes
                    .iter()
                    .map(|c| c.to_byte(ctx, private_key))
                    .collect();
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

        let leaves: Vec<Leaf> = json["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|leaf| {
                let classes: Vec<ByteLWE> = leaf["classes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        ByteLWE::from_byte_trivially(c.as_u64().unwrap() as u8, ctx, public_key)
                    })
                    .collect();
                Leaf { classes }
            })
            .collect();

        let final_leaves: Vec<LWE> = json["final_leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| public_key.allocate_and_trivially_encrypt_lwe(c.as_u64().unwrap(), ctx))
            .collect();

        Tree {
            root,
            stages,
            leaves,
            depth: json["depth"].as_u64().unwrap(),
            n_classes: json["n_classes"].as_u64().unwrap(),
            final_leaves,
        }
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

    pub fn save_perf_to_file(
        &self,
        duration: Duration,
        dataset_name: &str,
        n_trees: u64,
        depth: u64,
    ) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open("perf.csv")
            .unwrap();

        writeln!(
            file,
            "{},{},{},{:?}",
            n_trees, depth, dataset_name, duration
        )
        .unwrap();
    }
}

impl ClearTree {
    pub fn from_json(json_data: &Value, ctx: &Context, public_key: &PublicKey) -> Self {
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
                    .map(|count| count.as_u64().unwrap())
                    .collect(),
                id: i as u64,
            })
            .collect();

        ClearTree {
            root,
            nodes,
            leaves,
            depth: json_data["depth"].as_u64().unwrap(),
            n_classes: json_data["n_classes"].as_u64().unwrap(),
        }
    }
}

impl ClearForest {
    pub fn load_from_file(filepath: &str, ctx: &Context, public_key: &PublicKey) -> Self {
        let mut file = File::open(filepath).expect("Unable to open file");
        let mut json_string = String::new();
        file.read_to_string(&mut json_string)
            .expect("Unable to read data");

        let json_data: Value = serde_json::from_str(&json_string).unwrap();
        let trees: Vec<ClearTree> = json_data["trees"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tree_json| ClearTree::from_json(tree_json, ctx, public_key))
            .collect();

        ClearForest { trees }
    }
}

#[cfg(test)]
mod tests {
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
        let n_trees = 5;
        let depth = 3;
        let n_classes = 2;
        let f = ctx.full_message_modulus() as u64;

        // Create a new forest
        let forest = Forest::new(n_trees, depth, n_classes, f, &public_key, &ctx);

        // Define a file path for saving the forest
        let filepath = "./src/comp_free/test_forest.json";

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

            for (original_leaf, loaded_leaf) in
                original_tree.leaves.iter().zip(loaded_tree.leaves.iter())
            {
                for (original_class, loaded_class) in
                    original_leaf.classes.iter().zip(loaded_leaf.classes.iter())
                {
                    assert_eq!(
                        original_class.to_byte(&ctx, &private_key),
                        loaded_class.to_byte(&ctx, &private_key)
                    );
                }
            }
        }
    }

    #[test]
    fn test_clear_forest_from_file() {
        let filepath = "./src/comp_free/test_forest.json";
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(ctx.parameters());
        let public_key = &private_key.public_key;
        let clear_forest = ClearForest::load_from_file(filepath, &ctx, &public_key);

        assert_eq!(clear_forest.trees.len(), 5);
        assert_eq!(clear_forest.trees[0].depth, 3);
        assert_eq!(clear_forest.trees[0].n_classes, 2);
        assert_eq!(clear_forest.trees[0].leaves[0].counts[0], 1);
        assert_eq!(clear_forest.trees[0].leaves[0].counts[1], 0);

        assert_eq!(clear_forest.trees[0].root.threshold, 97);
        assert_eq!(clear_forest.trees[0].root.feature_index, 10);

        assert_eq!(clear_forest.trees[0].nodes[0][0].threshold, 474);
        assert_eq!(clear_forest.trees[0].nodes[0][0].feature_index, 3);
        assert_eq!(clear_forest.trees[0].nodes[0][0].id, 0);

        assert_eq!(clear_forest.trees[0].nodes[0][1].threshold, 1108);
        assert_eq!(clear_forest.trees[0].nodes[0][1].feature_index, 11);
        assert_eq!(clear_forest.trees[0].nodes[0][1].id, 1);

        
        
    }
}
