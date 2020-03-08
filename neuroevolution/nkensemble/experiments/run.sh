make -C ../nnet2nk_alg1 && cp ../nnet2nk_alg1/nnet2nke .
make -C ../nkdynp_alg2 && cp ../nkdynp_alg2/nkdynpe .
make -C ../NNcommittee_alg3 && cp ../NNcommittee_alg3/NNcommitteee .

cp ../nkdynp_alg2/EfficientHillClimbers-0.0.1-SNAPSHOT-r2014-11-26-181215.jar .

mkdir data
mkdir results

./nnet2nke 100  2  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k2
mv ./data/*.dat ./results/k2

./nnet2nke 100  3  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k3
mv ./data/*.dat ./results/k3

./nnet2nke 100  4  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k4
mv ./data/*.dat ./results/k4

./nnet2nke 100  5  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k5
mv ./data/*.dat ./results/k5

./nnet2nke 100  6  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k6
mv ./data/*.dat ./results/k6

./nnet2nke 100  7  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k7
mv ./data/*.dat ./results/k7

./nnet2nke 100  8  1 1 60
./nkdynpe
./NNcommitteee
mkdir results/k8
mv ./data/*.dat ./results/k8