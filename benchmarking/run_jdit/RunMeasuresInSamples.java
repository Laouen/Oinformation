import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import infodynamics.measures.continuous.kraskov.MultiInfoCalculatorKraskov1;
import infodynamics.measures.continuous.kraskov.DualTotalCorrelationCalculatorKraskov;
import infodynamics.measures.continuous.kraskov.SInfoCalculatorKraskov;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RunMeasuresInSamples {

    public static void main(String[] args) {

        try {
            String numpyFilesDir = args[0];
            String outputPath = args[1];
            int windowSize = 10000;
            
            String[] columns = {
                "distribution", "alpha", "window",
                "O-information",
                "Total Correlation",
                "Dual Total Correlation",
                "S-information"
            };

            System.out.println("Parameter: ");
            System.out.println("outputPath: " + outputPath);
            System.out.println("numpyFilesDir: " + numpyFilesDir);
            System.out.println("windowSize: " + windowSize);

            String[] systems = {
                "hh_normal",    "tt_normal",    "joint_normal",
                "hh_beta",      "tt_beta",      "joint_beta",
                "hh_exp",       "tt_exp",       "joint_exp",
                "hh_uniform",   "tt_uniform",   "joint_uniform"
            };
            String[] alphas = {"0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"};

            List<List<String>> rows = new ArrayList<List<String>>();

            // iterate over system and alpha
            for (int s = 0; s < systems.length; s++) {
                for (int a = 0; a < alphas.length; a++) {

                    String system = systems[s];
                    String alpha = alphas[a];
                    
                    // create the file path as a string with format numpyFilesDir/{system}_alpha-{alpha.npy
                    String filePath = String.format("%s/%s_alpha-%s.csv", numpyFilesDir, system, alpha);

                    // Load the numpy file
                    double[][] numpyData = CSVReader.readCSVFile(filePath);
                    int totalSamples = numpyData.length;
                    int nVariables = numpyData[0].length;

                    // print shape of data
                    System.out.println("\tData shape: " + numpyData.length + " x " + numpyData[0].length);

                    // get total windows as integer division
                    int nWindows = totalSamples / windowSize;

                    for (int w = 0; w < nWindows; w++) {

                        // Print current system, alpha, and window
                        System.out.println("\tSystem: " + system + ", Alpha: " + alpha + ", Window: " + w);

                        // Process data in windows of windowSize
                        int startIdx = w * windowSize;
                        int endIdx = Math.min(startIdx + windowSize, totalSamples);
                        if (endIdx - startIdx < windowSize) {
                            break; // not enough data for another full window
                        }

                        // Print start and end index
                        System.out.println("\tStart index: " + startIdx + ", End index: " + endIdx);

                        double[][] windowData = extractWindowData(numpyData, startIdx, endIdx);

                        // print shape of window data
                        System.out.println("\tWindow data shape: " + windowData.length + " x " + windowData[0].length);

                        // Compute O-information
                        OInfoCalculatorKraskov oInfoCalc = new OInfoCalculatorKraskov();
                        oInfoCalc.initialise(nVariables);
                        oInfoCalc.setObservations(windowData);
                        double oinfo_val = oInfoCalc.computeAverageLocalOfObservations();

                        // Compute the TC using the MultiInfoCalculatorKraskov1
                        MultiInfoCalculatorKraskov1 tcCalc = new MultiInfoCalculatorKraskov1();
                        tcCalc.initialise(nVariables);
                        tcCalc.setObservations(windowData);
                        double tc_val = tcCalc.computeAverageLocalOfObservations();

                        // Compute the DTC using the DualTotalCorrelationCalculatorKraskov
                        DualTotalCorrelationCalculatorKraskov dtcCalc = new DualTotalCorrelationCalculatorKraskov();
                        dtcCalc.initialise(nVariables);
                        dtcCalc.setObservations(windowData);
                        double dtc_val = dtcCalc.computeAverageLocalOfObservations();

                        // Compute the S-information using the SInfoCalculatorKraskov
                        SInfoCalculatorKraskov sinfoCalc = new SInfoCalculatorKraskov();
                        sinfoCalc.initialise(nVariables);
                        sinfoCalc.setObservations(windowData);
                        double sinfo_val = sinfoCalc.computeAverageLocalOfObservations();

                        List<String> row = new ArrayList<String>();
                        row.add(systems[s]);
                        row.add(alphas[a]);
                        row.add(Integer.toString(w));
                        row.add(Double.toString(oinfo_val));
                        row.add(Double.toString(tc_val));
                        row.add(Double.toString(dtc_val));
                        row.add(Double.toString(sinfo_val));
                        rows.add(row);
                    }
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

    private static double[][] extractWindowData(double[][] data, int startIdx, int endIdx) {
        int nVariables = data[0].length;
        double[][] windowData = new double[endIdx - startIdx][nVariables];
        for (int i = 0; i < nVariables; i++) {
            for (int j = startIdx; j < endIdx; j++) {
                windowData[j - startIdx][i] = data[j][i];
            }
        }
        return windowData;
    }
}
