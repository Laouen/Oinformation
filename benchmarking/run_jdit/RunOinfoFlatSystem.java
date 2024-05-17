import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class RunOinfoFlatSystem {

    public static void main(String[] args) {

        try {
            String outputPath = args[0];
            int T = 10000;
            int nRepeat = 20;
            String[] columns = {"n-plet", "method", "O-information", "alpha", "beta", "gamma"};

            System.out.println("Parameter: ");
            System.out.println("outputPath: " + outputPath);
            System.out.println("T: " + T);
            System.out.println("nRepeat: " + nRepeat);
        
            String[][] npletas = {
                // without synergistic and redundant source
                {"X1","X2","X3"},
                {"X1","X2","X3","X4"},
                {"X1","X2","X3","X4", "X5"},
                {"X1","X2","X3","X4", "X5", "X6"},

                // with synergistic and redundant source
                //{"Z00","Z01","X1","X2","X3"},
                //{"Z00","Z01","X1","X2","X3","X4"},
                //{"Z00","Z01","X1","X2","X3","X4", "X5"},
                {"Z00","X1","X2","X3","X4", "X5", "X6"},
                {"Z01","X1","X2","X3","X4", "X5", "X6"},
                {"Z00","Z01","X1","X2","X3","X4", "X5", "X6"}
            };
            double[] alphaValues = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
            double[] betaValues = alphaValues; // Same as alpha values for simplicity

            List<List<String>> rows = new ArrayList<List<String>>();
            
            for (double alpha : alphaValues) {
                for (double beta : betaValues) {
                    
                    System.out.println("Processing alpha=" + alpha + ", beta=" + beta);

                    double[][] npletOInfos = new double[npletas.length][nRepeat];
                    for (int j = 0; j < nRepeat; j++) {

                        // Generate random system
                        RandomSystemsGenerator.RandomSystem<String> system = RandomSystemsGenerator.generateFlatSystem(alpha, beta, 0.1, T);
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

                        List<String> row = Arrays.asList(
                            RandomSystemsGenerator.getNpletName(nplet),
                            "JDIT",
                            String.valueOf(nplet_oinfo),
                            String.valueOf(alpha),
                            String.valueOf(beta),
                            String.valueOf(0.1)
                        );

                        rows.add(row);
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
