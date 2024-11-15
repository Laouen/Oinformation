
# Compile and run

## RunMeasureTimesJDIT
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunMeasureTimesJDIT.java
java -cp .:infodynamics.jar RunMeasureTimesJDIT ../results/times/library-jdit_estimator-ksg.tsv

## RunOinfoReLUSytem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoReLUSystem.java
java -cp .:infodynamics.jar RunOinfoReLUSystem ../results/o_info/system-relu_pow-0.5_repeat-20_t-10000_JDIT.tsv 0.5
java -cp .:infodynamics.jar RunOinfoReLUSystem ../results/o_info/system-relu_pow-1.0_repeat-20_t-10000_JDIT.tsv 1.0

## RunOinfoSmoothSoftReLUSytem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoSmoothSoftReLUSystem.java
java -cp .:infodynamics.jar RunOinfoSmoothSoftReLUSystem ../results/o_info/system-smoothSoftRelu_pow-0.5_repeat-20_t-10000_JDIT.tsv 0.5

## RunOinfoXORSytem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoXORSystem.java
java -cp .:infodynamics.jar RunOinfoXORSystem ../results/o_info/system-xor_repeat-20_t-10000_JDIT.tsv

## RunOinfoFlatSystem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoFlatSystem.java
java -cp .:infodynamics.jar RunOinfoFlatSystem ../results/o_info/system-flat_repeat-20_t-10000_JDIT.tsv 0.1

## RunMeasuresInSamples
javac -cp infodynamics.jar TSVWriter.java CSVReader.java RunMeasuresInSamples.java
java -cp .:infodynamics.jar RunMeasuresInSamples /home/laouen.belloli/Documents/data/Oinfo/PGM_data /home/laouen.belloli/Documents/git/Oinformation/benchmarking/results/pgm/pgm_results_jidt.tsv

## RunOinfoTimeBySampleSize
javac -cp infodynamics.jar TSVWriter.java CSVReader.java RunOinfoTimeBySampleSize.java
java -cp .:infodynamics.jar RunOinfoTimeBySampleSize /home/laouen.belloli/Documents/data/Oinfo/random_sample_sizes /home/laouen.belloli/Documents/git/Oinformation/benchmarking/results/times/by_sample_size_library-jidt.tsv
