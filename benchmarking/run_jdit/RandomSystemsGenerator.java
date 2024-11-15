import java.util.Random;
import java.util.Arrays;
import java.util.List;
import java.lang.Math;

public class RandomSystemsGenerator {

    public static class RandomSystem<T> {
        private double[][] data; // (n_samples, n_variables)
        public final T[] columns;

        public RandomSystem(double[][] data, T[] columns) {
            this.data = data;
            this.columns = columns;
        }

        public double[][] getNPletData(T[] nplet) {

            int N = nplet.length;
            int T = this.data.length;

            double[][] result = new double[T][N];

            List<T> nplet_list = Arrays.asList(nplet);

            // For each variable i
            for (int j = 0; j < N; j++) {
                if (nplet_list.contains(this.columns[j])) {
                    for (int i = 0; i < T; i++) {
                        result[i][j] = this.data[i][j];
                    }
                }
            }

            return result;
        }


    };

    public static String getNpletName(String[] variables) {
            
        String nplet_name = variables[0];

        for (int i = 1; i < variables.length-1; i++) {
            nplet_name = nplet_name + "-" + variables[i];
        }
        
        return nplet_name + "-" + variables[variables.length-1];
    };

    public static RandomSystem<String> generateReLUSystem(double alpha, double beta, double powFactor, int T) {
        if (!(0 <= alpha && alpha <= 1.0) || !(0 <= beta && beta <= 1.0)) {
            throw new IllegalArgumentException("alpha and beta must be in range [0,1]");
        }

        Random random = new Random();
        double[][] data = new double[T][4];  // 4 rows for X1, X2, Z_syn, Z_red

        for (int i = 0; i < T; i++) {
            double Z_syn = random.nextGaussian();
            double Z_red = random.nextGaussian();

            data[i][0] = alpha * Math.pow(Math.max(Z_syn, 0), powFactor) + beta * Z_red; // X1
            data[i][1] = -alpha * Math.pow(Math.max(-Z_syn, 0), powFactor) + beta * Z_red; // X2
            data[i][2] = Z_syn; // Z_syn
            data[i][3] = Z_red; // Z_red
        }

        String[] columns = {"X1", "X2", "Z_syn", "Z_red"};
        RandomSystem<String> result = new RandomSystem<String>(data, columns);
        return result;
    };

    public static float smoothSoftPlus(double X) {
        // Compute the smooth softplus function
        return (float)Math.log1p(Math.exp(X-3.0));
    }

    public static RandomSystem<String> generateSmoothSoftReLUSystem(double alpha, double beta, double powFactor, int T) {
        if (!(0 <= alpha && alpha <= 1.0) || !(0 <= beta && beta <= 1.0)) {
            throw new IllegalArgumentException("alpha and beta must be in range [0,1]");
        }

        Random random = new Random();
        double[][] data = new double[T][4];  // 4 rows for X1, X2, Z_syn, Z_red

        for (int i = 0; i < T; i++) {
            double Z_syn = random.nextGaussian();
            double Z_red = random.nextGaussian();

            data[i][0] = alpha * Math.pow(smoothSoftPlus(Z_syn), powFactor) + beta * Z_red; // X1
            data[i][1] = -alpha * Math.pow(smoothSoftPlus(-Z_syn), powFactor) + beta * Z_red; // X2
            data[i][2] = Z_syn; // Z_syn
            data[i][3] = Z_red; // Z_red
        }

        String[] columns = {"X1", "X2", "Z_syn", "Z_red"};
        RandomSystem<String> result = new RandomSystem<String>(data, columns);
        return result;
    };

    public static RandomSystem<String> generateXORSystem(double alpha, int T) {
        if (!(0 <= alpha && alpha <= 1.0)) {
            throw new IllegalArgumentException("alpha and beta must be in range [0,1]");
        }

        Random random = new Random();
        double[][] data = new double[T][3];  // 3 columns for X1, X2, Zxor

        for (int i = 0; i < T; i++) {
            data[i][0] = random.nextGaussian(); // X1
            data[i][1] = random.nextGaussian(); // X2
            data[i][2] = random.nextGaussian(); // Z

            // if (X1 xor X2)
            if ((data[i][0] > 0) ^ (data[i][1] > 0)) {
                // Z_xor = alpha*(Z + 4*XOR) + (1-alpha)*Z
                data[i][2] = alpha*(data[i][2] + 4) + (1-alpha)*data[i][2];
            }
        }

        String[] columns = {"X1", "X2", "Zxor"};
        RandomSystem<String> result = new RandomSystem<String>(data, columns);
        return result;
    };

    public static RandomSystem<String> generateFlatSystem(double alpha, double beta, double gamma, int T) {

        if (!(0 <= alpha && alpha <= 1.0) || !(0 <= beta && beta <= 1.0)) {
            throw new IllegalArgumentException("alpha and beta must be in range [0,1]");
        }

        Random random = new Random();
        double[][] data = new double[T][8];  // 4 columns for X1, ..., X6, Z00, Z01

        for (int i = 0; i < T; i++) {

            double Z00 = random.nextGaussian();
            double Z01 = random.nextGaussian();

            data[i][0] = gamma*random.nextGaussian() + beta*Z01 + alpha*Math.log(Math.abs(Z00)); // X1
            data[i][1] = gamma*random.nextGaussian() + beta*Z01 + alpha*Z00; // X2
            data[i][2] = gamma*random.nextGaussian() + beta*Z01 + alpha*Math.pow(Z00,2); // X3
            data[i][3] = gamma*random.nextGaussian() + beta*Z01 + alpha*Math.exp(Z00); // X4
            data[i][4] = gamma*random.nextGaussian() + beta*Z01 + alpha*Math.sin(Z00); // X5
            data[i][5] = gamma*random.nextGaussian() + beta*Z01 + alpha*Math.cos(Z00); // X6

            data[i][6] = Z00;
            data[i][7] = Z01;
        }

        String[] columns = {"X1", "X2", "X3", "X4", "X5", "X6", "Z00", "Z01"};
        RandomSystem<String> result = new RandomSystem<String>(data, columns);
        return result;
    };

    public static RandomSystem<Integer> generateMultivariateNormal(int T, int N) {

        Random random = new Random();
        double[][] data = new double[T][N];

        for (int i = 0; i < T; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] = random.nextGaussian();
            }
        }

        Integer[] columns = new Integer[N];
        for (int i = 0; i < N; i++) {
            columns[i] = i;
        }

        RandomSystem<Integer> result = new RandomSystem<Integer>(data, columns);
        return result;
    };
}
