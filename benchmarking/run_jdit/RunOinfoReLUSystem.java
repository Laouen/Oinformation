import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import java.util.Arrays;

public class RunOinfoReLUSystem {

    public static void main(String[] args) {

        try {
            String outputPath = "./results.tsv";
            double powFactorValue = 0.5;
            int T = 10000;
            int nRepeat = 20;
            String[] columns = {"n-plet", "method", "alpha", "beta", "O-information"};

            System.out.println("Parameter: ");
            System.out.println("outputPath: " + outputPath);
            System.out.println("powFactorValue: " + powFactorValue);
            System.out.println("T: " + T);
            System.out.println("nRepeat: " + nRepeat);
        
            String[][] npletas = {
                {"X1", "X2", "Z_syn"},
                {"X1", "X2", "Z_syn", "Z_red"}
            };
            double[] alphaValues = {
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
            };
            double[] betaValues = alphaValues; // Same as alpha values for simplicity

            String[][] rows = new String[alphaValues.length*betaValues.length*npletas.length][5];

            int a = 0;
            for (double alpha : alphaValues) {
                for (double beta : betaValues) {
                    
                    System.out.println("Processing alpha=" + alpha + ", beta=" + beta);

                    double[][] npletOInfos = new double[npletas.length][nRepeat];
                    for (int j = 0; j < nRepeat; j++) {

                        // Generate random system
                        RandomSystemsGenerator.RandomSystem system = RandomSystemsGenerator.generateReluSystem(alpha, beta, powFactorValue, T);
                        for(int i = 0; i < npletas.length; i++) {

                            String[] nplet = npletas[i];

                            // Extract data from the desired nplet
                            double[][] data = system.getNPletData(nplet);

                            // Compute O-information
                            OInfoCalculatorKraskov oInfoCalc = new OInfoCalculatorKraskov();
                            oInfoCalc.initialise(data[0].length);
                            oInfoCalc.setObservations(data);
                            npletOInfos[i][j] = oInfoCalc.computeAverageLocalOfObservations();
                        }
                    }

                    // Save final repeat resuls into the 
                    for (int i = 0; i < npletas.length; i++) {

                        String[] nplet = npletas[i];

                        double nplet_oinfo = Arrays.stream(npletOInfos[i]).average().getAsDouble();

                        rows[a][0] = RandomSystemsGenerator.getNpletName(nplet);
                        rows[a][1] = "JDIT";
                        rows[a][2] = String.valueOf(alpha);
                        rows[a][3] = String.valueOf(beta);
                        rows[a][4] = String.valueOf(nplet_oinfo);
                        a = a + 1;
                    }
                }
            }

            TSVWriter.writeToTSV(rows, outputPath, columns);
        } catch (Exception e) {
            System.err.println("Error somewhere: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
