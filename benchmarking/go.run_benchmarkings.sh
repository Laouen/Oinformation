#######################################
####   Run O information analysis  ####
#######################################

# Generate a flat systems and run the o information
python run_oinfo_flat_system.py --output_path ./results/o_info/system-flat_repeat-20_t-10000.tsv --n_repeat 20 --T 10000

# Generate a relu systems and run the o information
python run_oinfo_relu_system.py --output_path ./results/o_info/system-relu_pow-0.5_repeat-20_t-10000.tsv --pow_factor 0.5 --n_repeat 20 --T 10000
python run_oinfo_relu_system.py --output_path ./results/o_info/system-relu_pow-1.0_repeat-20_t-10000.tsv --pow_factor 1.0 --n_repeat 20 --T 10000

# Generate a xor systems and run the o information
python run_oinfo_xor_system.py --output_path ./results/o_info/system-xor_repeat-20_t-10000.tsv --n_repeat 20 --T 10000

#######################################
### Run entropy estimators analysis ###
#######################################

python run_estimators_error.py --output_path ./results/estimators/distribution-gaussian_repeat-20.tsv --distribution gaussian --n_repeat 20
python run_estimators_error.py --output_path ./results/estimators/distribution-uniform_repeat-20.tsv --distribution uniform --n_repeat 20
python run_estimators_error.py --output_path ./results/estimators/distribution-dirichlet_repeat-20.tsv --distribution dirichlet --n_repeat 20

#######################################
#### Run computation time analysis ####
#######################################

python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 5 --step_N 5 --max_N 30 \
    --min_bs 100000 --step_bs 100000 --max_bs 1000000 \
    --min_order 3 --max_order 30 --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device.tsv

python run_measure_times_HOI.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --estimator gcmi \
    --output_path ./results/times/library-hoi_estimator-gc.tsv

python run_measure_times_HOI.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --estimator lin_est \
    --output_path ./results/times/library-hoi_estimator-linest.tsv

python run_measure_times_GCMI_NPEET.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --library GCMI \
    --output_path ./results/times/library-gcmi_estimator-gc.tsv

python run_measure_times_GCMI_NPEET.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --library NPEET \
    --output_path ./results/times/library-npeet_estimator-ksg.tsv


########################################################
#### Run computation time per sample size analysis #####
########################################################

python run_oino_time_by_sample_size.py \
    --files_dir /home/laouen.belloli/Documents/data/Oinfo/random_sample_sizes \
    --output_path /home/laouen.belloli/Documents/git/Oinformation/benchmarking/results/times/by_sample_size_library-thoi.tsv 