python run_THOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gc \
    --output_path ./results/thoi_gc_times.tsv

python run_HOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gcmi \
    --output_path ./results/hoi_gcmi_times.tsv

python run_HOI.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator lin_est \
    --output_path ./results/hoi_linest_times.tsv

python run_GCMI_NPEET.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator gcmi \
    --output_path ./results/gcmi_times.tsv

python run_GCMI_NPEET.py \
    --min_T 1000 --step_T 100000 --max_T 1000000 \
    --min_N 50 --step_N 5 --max_N 100 \
    --min_order 3 --max_order 20 \
    --estimator npeet \
    --output_path ./results/npeet_times.tsv