declare -A experiments

# number threads is auto all available threads, can be overwritten with -t
# options="-t 40"
options=""

## PASTE EXPERIMENTS HERE
experiments["example"]="-r 1 -s 0.1 --seed 42 --step-size 2 -i 20"
#experiments["32-low-masked-20"]="-r 20 -n 32 --type-nonzeros rearranged --seed 3629895 -s 1.7 1.8 1.9"
#experiments["32-high-masked-5"]="-r 5 -n 32 --type-nonzeros rearranged --seed 362989 -s 2.9 3.0 3.1"
## END PASTE EXPERIMENTS HERE

# create folders
for val in ${!experiments[*]}; do
  mkdir -p "runs/$val"
done

# make sure that the current bp rust code is built
make

# run experiments and save results
for ix in ${!experiments[*]}; do
  echo $ix
  echo ${experiments[$ix]} > results_cmd.txt
  echo $options >> results_cmd.txt
#  git rev-parse HEAD > results_gitcommit.txt
#  (cd belief_propagation && git rev-parse HEAD) > results_gitcommit_submodule.txt
  python -u python/test.py ${experiments[$ix]} $options | tee results.txt
  mv results* "runs/$ix"
  tar -czf "runs/$ix.tar.gz" -C "runs/$ix" .
done
