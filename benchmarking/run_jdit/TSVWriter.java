import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TSVWriter {

    public static void writeToTSV(String[][] data, String outputPath, String[] columns) throws IOException {
        // Create a BufferedWriter instance to write to the specified output path
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            // Write column headers separated by tabs
            for (int i = 0; i < columns.length; i++) {
                writer.write(columns[i]);
                if (i < columns.length - 1) {
                    writer.write("\t");  // Tab-separated
                }
            }
            writer.newLine();  // Move to the next line after headers

            // Write data rows
            for (String[] row : data) {
                for (int j = 0; j < row.length; j++) {
                    writer.write(row[j]);
                    if (j < row.length - 1) {
                        writer.write("\t");  // Tab-separated
                    }
                }
                writer.newLine();  // Move to the next line after each row
            }
        }
    }
}