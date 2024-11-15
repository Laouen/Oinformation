#######################################
####   Run O information analysis  ####
#######################################

# Generate a flat systems and run the o information
python run_oinfo_flat_system.py --output_path ./results/o_info/system-flat_repeat-20_t-10000.tsv --n_repeat 20 --T 10000

# Generate a relu systems and run the o information
python run_oinfo_relu_system.py \
    --output_path ./results/o_info/system-relu_pow-0.5_repeat-20_t-10000.tsv \
    --pow_factor 0.5 \
    --n_repeat 20 \
    --T 10000

python run_oinfo_relu_system.py \
    --output_path ./results/o_info/system-relu_pow-1.0_repeat-20_t-10000.tsv \
    --pow_factor 1.0 \
    --n_repeat 20 \
    --T 10000

# Generate a xor systems and run the o information
python run_oinfo_xor_system.py --output_path ./results/o_info/system-xor_repeat-20_t-10000.tsv --n_repeat 20 --T 10000



python create_smoothSoftReLU_dataset.py \
    --output_path ./results/data \
    --pow_factor 0.5 \
    --n_repeat 20 \
    --T 10000

python run_oinfo_relu_system.py \
    --output_path ./results/o_info/system-smoothSoftRelu_pow-0.5_repeat-20_t-10000.tsv \
    --pow_factor 0.5 \
    --n_repeat 20 \
    --T 10000




#######################################
####    Run Simulated Annealing    ####
#######################################

python run_simulated_annealing_multi_order.py \
    --path_covariance_matrix /home/laouen.belloli/Documents/data/Oinfo/tt_hh/N-100_example_covmat.npy \
    --output_path ./results/simulated_annealing \
    --repeat 2000

#######################################
### Run entropy estimators analysis ###
#######################################

python run_estimators_error.py --output_path ./results/estimators/distribution-gaussian_repeat-20.tsv --distribution gaussian --n_repeat 20
python run_estimators_error.py --output_path ./results/estimators/distribution-uniform_repeat-20.tsv --distribution uniform --n_repeat 20
python run_estimators_error.py --output_path ./results/estimators/distribution-dirichlet_repeat-20.tsv --distribution dirichlet --n_repeat 20

#######################################
#### Run computation time analysis ####
#######################################

#--min_bs 100000 --step_bs 100000 --max_bs 1000000 \
python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 30 \
    --min_bs 100000 --step_bs 150000 --max_bs 251000 \
    --min_order 3 --max_order 31 \
    --indexing_method indexes --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device-cpu_indexing-indexes_bs_10_25.tsv

python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 30 \
    --min_bs 100000 --step_bs 150000 --max_bs 251000 \
    --min_order 3 --max_order 30 \
    --indexing_method hot_encoded --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device-cpu_indexing-hotencoded.tsv

python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 5 \
    --min_bs 100000 --step_bs 100000 --max_bs 1000000 \
    --min_order 3 --max_order 30 \
    --indexing_method indexes \
    --output_path ./results/times/library-thoi_estimator-gc_device-cuda_indexing-indexes.tsv

python run_measure_times_HOI_v2.py \
    --min_T 1000 \
    --min_N 30 \
    --min_order 3 --max_order 30 \
    --output_path ./results/times/library-hoiv2_estimator-gc.tsv

python run_measure_times_HOI_v1.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --estimator gcmi \
    --output_path ./results/times/library-hoiv1_estimator-gc.tsv

python run_measure_times_HOI_v1.py \
    --min_T 1000 --min_N 30 \
    --min_order 3 --max_order 30 \
    --estimator lin_est \
    --output_path ./results/times/library-hoiv1_estimator-linest.tsv

python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 5 \
    --min_bs 100000 --step_bs 150000 --max_bs 251000 \
    --min_order 3 --max_order 31 \
    --indexing_method indexes --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device-cpu_indexing-indexes_bs_10_25.tsv

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





############ TEMPORAL ####################

python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 30 --step_N 5 --max_N 31 \
    --min_bs 200000 --step_bs 100000 --max_bs 2000001 \
    --min_order 15 --max_order 31 \
    --indexing_method hot_encoded --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device-cpu_indexing-hotencoded_new1.tsv


python run_measure_times_THOI.py \
    --min_T 1000 \
    --min_N 30 --step_N 5 --max_N 31 \
    --min_bs 300000 --step_bs 100000 --max_bs 1000000 \
    --min_order 3 --max_order 30 \
    --indexing_method hot_encoded --use_cpu \
    --output_path ./results/times/library-thoi_estimator-gc_device-cpu_indexing-hotencoded_new2.tsv



############### RUN ANESTHESIA ####################

python run_anesthesia.py \
    --input_path /home/laouen.belloli/Documents/data/Oinfo/fmri_anesthesia/42003_2023_5063_MOESM3_ESM/nets_by_subject \
    --output_path ./results/anesthesia/ \
    --func effect_size

python run_anesthesia.py \
    --input_path /home/laouen.belloli/Documents/data/Oinfo/fmri_anesthesia/42003_2023_5063_MOESM3_ESM/nets_by_subject \
    --output_path ./results/anesthesia/ \
    --func roc_auc

############### RUN MULTIPLE DATASETS ####################

python run_multiple_datasets_vs_one_times.py \
    --min_D 21 --step_D 1 --max_D 40 \
    --min_T 10000 \
    --min_N 20 \
    --min_bs 10000 \
    --indexing_method indexes --device cpu \
    --output_path ./results/times/single_vs_listed_datasets_21-40.tsv