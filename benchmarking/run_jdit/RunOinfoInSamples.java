import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RunOinfoInSamples {

    public static void main(String[] args) {

        try {
            String numpyFilesDir = args[0];
            String outputPath = args[1];
            int windowSize = 10000;
            
            String[] columns = {"distribution", "alpha", "window", "O-information"};

            System.out.println("Parameter: ");
            System.out.println("outputPath: " + outputPath);
            System.out.println("numpyFilesDir: " + numpyFilesDir);
            System.out.println("windowSize: " + windowSize);

            //String[] systems = {"db", "db1", "db2"}; 
            //String[] systems = {"hh_normal","tt_normal","hh_beta","tt_beta","hh_exp","tt_exp","hh_uniform","tt_uniform","db"};
            String[] systems = {"joint_beta","joint_exp","joint_uniform"};
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
                    double[][] numpyData = readCSVFile(filePath);
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

                        List<String> row = new ArrayList<String>();
                        row.add(systems[s]);
                        row.add(alphas[a]);
                        row.add(Integer.toString(w));
                        row.add(Double.toString(oinfo_val));
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

    private static double[][] readCSVFile(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    // print values[i]
                    //System.out.println("values[i]: " + values[i]);
                    row[i] = Double.parseDouble(values[i]);
                }
                data.add(row);
            }
        }
        // Convert List<double[]> to double[][]
        double[][] array = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            array[i] = data.get(i);
        }
        return array;
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
