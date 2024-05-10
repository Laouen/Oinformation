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

python run_estimators_error.py --output_path ./results/estimators_error_repeat-20.tsv --n_repeat 20

#######################################
#### Run computation time analysis ####
#######################################

python run_measure_times_THOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gc \
    --output_path ./results/times/library-thoi_estimator-gc.tsv

python run_measure_times_HOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gcmi \
    --output_path ./results/times/library-hoi_estimator-gc.tsv

python run_measure_times_HOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator lin_est \
    --output_path ./results/times/library-hoi_estimator-linest.tsv

python run_measure_times_GCMI_NPEET.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gcmi \
    --output_path ./results/times/library-gcmi_estimator-gc.tsv

python run_measure_times_GCMI_NPEET.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator npeet \
    --output_path ./results/times/library-npeet_estimator-knn.tsv