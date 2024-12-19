// use crate::clear_model::{generate_clear_random_tree, ClearDataset, ClearTree};

// fn example_clear_training() {
//     let mut clear_dataset = ClearDataset::from_file("data/iris_2bits.csv".to_string());

//     // get the train and test datasets
//     let (train_dataset, test_dataset) = clear_dataset.split(0.8);
//     let column_domains = clear_dataset.column_domains.clone();
//     let n_classes = column_domains[column_domains.len() - 1].1 + 1;
//     let n_trees = 1;

//     let mut forest: Vec<ClearTree> = Vec::new();

//     // Training the forest
//     for i in 0..n_trees {
//         let mut clear_tree =
//             generate_clear_random_tree(4, n_classes, column_domains.clone(), train_dataset.f);
//         train_dataset.records.iter().for_each(|record| {
//             clear_tree.update_statistic(record.to_vec());
//         });

//         clear_tree.assign_label_to_leafs();
//         clear_tree.print_tree();
//         forest.push(clear_tree);
//     }

//     // Testing the forest
//     let mut correct = 0;
//     let mut total = 0;
//     for record in test_dataset.records.iter() {
//         let mut votes = vec![0; n_classes as usize];
//         for tree in forest.iter() {
//             let label = tree.infer_label(record.to_vec());
//             votes[label as usize] += 1;
//         }
//         let predicted_label = votes.iter().enumerate().max_by_key(|x| x.1).unwrap().0 as u64;
//         let true_label = record[record.len() - 1];
//         if predicted_label == true_label {
//             correct += 1;
//         }
//         total += 1;
//     }
//     print!("\nNumber of trees : {}\n", n_trees);
//     print!("Size of training dataset : {}\n", train_dataset.n);
//     print!("Size of testing dataset : {}\n", test_dataset.n);
//     println!("Accuracy: {}", correct as f64 / total as f64);
// }
