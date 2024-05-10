import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Iterator;

public class TSVWriter {

    public static void writeToTSV(List<List<String>> data, String outputPath, String[] columns) throws IOException {
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
            for (List<String> row : data) {
                Iterator<String> it = row.iterator();
                while (it.hasNext()) {
                    writer.write(it.next());
                    if (it.hasNext()) {
                        writer.write("\t");  // Tab-separated
                    }
                }
                writer.newLine();  // Move to the next line after each row
            }
        }
    }
}