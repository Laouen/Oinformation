import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RunOinfoTimeBySampleSize {

    public static void main(String[] args) {

        try {
            String numpyFilesDir = args[0];
            String outputPath = args[1];
            
            String[] columns = {"sample size", "iteration", "time"};

            System.out.println("Parameter: ");
            System.out.println("outputPath: " + outputPath);
            System.out.println("numpyFilesDir: " + numpyFilesDir);

            String[] sample_sizes = {
                "1000","5000","10000", "15000", "20000", "25000", "30000", "35000", "40000", "45000", "50000",
                "55000", "60000", "65000", "70000", "75000", "80000", "85000", "90000", "95000", "100000"
            };
            
            List<List<String>> rows = new ArrayList<List<String>>();

            // iterate over system and alpha
            for (int s = 0; s < sample_sizes.length; s++) {
                String sample_size = sample_sizes[s];
                // create the file path as a string with format numpyFilesDir/{system}_alpha-{alpha.npy
                String filePath = String.format("%s/nsamples-%s_nvars-10.csv", numpyFilesDir, sample_size);

                // Load the numpy file
                double[][] numpyData = CSVReader.readCSVFile(filePath);
                int totalSamples = numpyData.length;
                int nVariables = numpyData[0].length;

                // print shape of data
                System.out.println("\tData shape: (n_samples = " + totalSamples + ", n_variables = " + numpyData[0].length + ")");

                for (int a = 0; a < 100; a++) {

                    long start = System.currentTimeMillis();

                    // Compute O-information
                    OInfoCalculatorKraskov oInfoCalc = new OInfoCalculatorKraskov();
                    oInfoCalc.initialise(nVariables);
                    oInfoCalc.setObservations(numpyData);
                    double oinfo_val = oInfoCalc.computeAverageLocalOfObservations();

                    long end = System.currentTimeMillis();
                    double timeElapsed = (double)(end - start) / 1000.0; //elapsed time in seconds

                    List<String> row = new ArrayList<String>();
                    row.add(Integer.toString(totalSamples));
                    row.add(Integer.toString(a));
                    row.add(Double.toString(timeElapsed));
                    rows.add(row);
                }

                // Print shape of rows
                System.out.println("##################");
                System.out.println("Rows shape: " + rows.size() + " x " + rows.get(0).size());
                System.out.println("##################");

                // Write the results incrementally to a tsv file
                TSVWriter.writeToTSV(rows, outputPath, columns);
            }

        } catch (Exception e) {
            System.err.println("Error somewhere: " + e.getMessage());
            e.printStackTrace();
        }
    }
}