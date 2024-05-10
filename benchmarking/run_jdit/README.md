
# Compile and run

## RunMeasureTimesJDIT
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunMeasureTimesJDIT.java
java -cp .:infodynamics.jar RunMeasureTimesJDIT ../results/o_info/library-jdit_estimator-knn.tsv

## RunOinfoReLUSytem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoReLUSystem.java
java -cp .:infodynamics.jar RunOinfoReLUSystem ../results/o_info/system-relu_pow-0.5_repeat-20_t-10000_JDIT.tsv 0.5
java -cp .:infodynamics.jar RunOinfoReLUSystem ../results/o_info/system-relu_pow-1.0_repeat-20_t-10000_JDIT.tsv 1.0

## RunOinfoFlatSystem
javac -cp infodynamics.jar RandomSystemsGenerator.java TSVWriter.java RunOinfoFlatSystem.java
java -cp .:infodynamics.jar RunOinfoFlatSystem ../results/o_info/system-flat_repeat-20_t-10000_JDIT.tsv 0.1
