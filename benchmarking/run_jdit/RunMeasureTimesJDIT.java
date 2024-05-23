import infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class RunMeasureTimesJDIT {

    private static void calculateOrderOinfo(RandomSystemsGenerator.RandomSystem<Integer> system, int order) {
        List<Integer> currentCombination = new ArrayList<>();
        combine(system, order, 0, currentCombination);
    }

    private static void combine(RandomSystemsGenerator.RandomSystem<Integer> system, int order, int start, List<Integer> current) {
        
        // Base case
        if (current.size() == order) {

            try {
                // Extract data from the desired nplet
                Integer[] nplet = current.toArray(new Integer[0]);

                double[][] data = system.getNPletData(nplet);

                // Compute O-information
                OInfoCalculatorKraskov oInfoCalc = new OInfoCalculatorKraskov();
                oInfoCalc.initialise(data[0].length);
                oInfoCalc.setObservations(data);
                oInfoCalc.computeAverageLocalOfObservations();
            } catch (Exception e) {
                System.err.println("Error somewcolumnshere: " + e.getMessage());
                e.printStackTrace();
            }
            return;
        }

        // Recursive step
        for (int i = start; i < system.columns.length; i++) {
            current.add(system.columns[i]);
            combine(system, order, i + 1, current);
            current.remove(current.size() - 1); // Backtrack
        }
    }

    public static void main(String[] args) {

        try {
            int min_T = 1000;
            int step_T = 100000;
            int max_T = 1001;

            int min_N = 30;
            int step_N = 5;
            int max_N = 31;

            int min_order = 3;
            int max_order = 30;

            String outputPath = args[0];
            String[] columns = {"library", "estimator", "T", "N", "order", "time"};

            List<List<String>> rows = new ArrayList<>();

            for (int T = min_T; T <= max_T; T = T + step_T) {
                for (int N = min_N; N <= max_N; N = N + step_N) {
                    
                    System.out.println("Processing T=" + T + ", N=" + N);
                    
                    RandomSystemsGenerator.RandomSystem<Integer> system = RandomSystemsGenerator.generateMultivariateNormal(T, N);
                    
                    for (int order = min_order; order <= max_order; order++) {

                        System.out.println("Processing T=" + T + ", N=" + N + ", order=" + order);

                        long start = System.currentTimeMillis();
                        calculateOrderOinfo(system, order);
                        long finish = System.currentTimeMillis();
                        long timeElapsed = (finish - start) / 1000; //elapsed time in seconds

                        rows.add(Arrays.asList(
                            "JIDT", "KSG",
                            String.valueOf(T), String.valueOf(N),
                            String.valueOf(order), String.valueOf(timeElapsed)
                        ));
                        TSVWriter.writeToTSV(rows, outputPath, columns);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error somewcolumnshere: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
