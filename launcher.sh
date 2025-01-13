
# Run the program with the following parameters:
# dataset-name: iris
# number-of-experiments: 100
# precisions: 2,3,4,5
# num-forests: 10,40,60
# quantizations: uni,non_uni,fd
# depths: 2,3,4
# Note : Only clear training can be used with the original datasets that contain continuous values (such as Iris)
# Adapt the main.



# cargo run --release -- --dataset-name=iris --number-of-experiments=100 --precisions=2,3,4,5 --num-forests=10,40,60 --quantizations=uni,non_uni,fd --depths=2,3,4
cargo run --release -- --dataset-name=iris --number-of-experiments=100 --precisions=2,3,4,5 --num-forests=10,40,60 --quantizations=uni,non_uni,fd --depths=2,3,4