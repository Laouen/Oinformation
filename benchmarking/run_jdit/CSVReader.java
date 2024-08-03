import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {
    public static double[][] readCSVFile(String filePath) throws IOException {
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
}
