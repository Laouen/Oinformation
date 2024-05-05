# Generate hierarchical and flat systems for different parameters of alfa, beta and gamma
python run_generate_systems.py --output_path ./results/systems --T 10000

# Generate hierarchical systems and run the o information
python run_hierarchical_system.py --output_path ./hierarchical_system.tsv --T 10000

# Generate a flat systems and run the o information
python run_flat_system.py --output_path ./hierarchical_system.tsv --T 10000