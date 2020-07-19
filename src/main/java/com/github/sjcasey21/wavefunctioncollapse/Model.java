package com.github.sjcasey21.wavefunctioncollapse;


import java.awt.image.BufferedImage;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

class StackEntry {
    private int x;
    private int y;

    public StackEntry(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getFirst() {
        return this.x;
    }

    public int getSecond() {
        return this.y;
    }
}

abstract class Model {
    protected boolean[][] wave;
    protected int[][][] propagator;
    int[][][] compatible;
    protected int[] observed;

    StackEntry[] stack;
    int stacksize;

    protected Random random;
    protected int FMX, FMY, T;
    protected boolean periodic;
    protected Double[] weights;
    double[] weightLogWeights;

    int[] sumsOfOnes;
    double sumOfWeights, sumOfWeightLogWeights, startingEntropy;
    double[] sumsOfWeights, sumsOfWeightLogWeights, entropies;

    protected Model(int width, int height) {
        this.FMX = width;
        this.FMY = height;
    }

    protected abstract boolean onBoundary(int x, int y);

    public abstract BufferedImage graphics();

    protected static int[] DX = {-1, 0, 1, 0};
    protected static int[] DY = {0, 1, 0, -1};
    static int[] oppposite = {2, 3, 0, 1};

    static int randomIndice(double[] arr, double r) {
        double sum = 0;

        for (int j = 0; j < arr.length; j++) sum += arr[j];

        for (int j = 0; j < arr.length; j++) arr[j] /= sum;

        int i = 0;
        double x = 0;

        while (i < arr.length) {
            x += arr[i];
            if (r <= x) return i;
            i++;
        }

        return 0;
    }

    public static long toPower(int a, int n) {
        long product = 1;
        for (int i = 0; i < n; i++) product *= a;
        return product;
    }

    void init() {
        wave = new boolean[FMX * FMY][];

        System.out.println("Wave.len: " + wave.length);

        compatible = new int[wave.length][][];

        for (int i = 0; i < wave.length; i++) {
            wave[i] = new boolean[T];
            compatible[i] = new int[T][];
            for (int t = 0; t < T; t++)
                compatible[i][t] = new int[4];
        }

        System.out.println("Wave allocated");


        weightLogWeights = new double[T];
        sumOfWeights = 0;
        sumOfWeightLogWeights = 0;

        for (int t = 0; t < T; t++) {
            weightLogWeights[t] = weights[t] * Math.log(weights[t]);
            sumOfWeights += weights[t];
            sumOfWeightLogWeights += weightLogWeights[t];
        }

        System.out.println("Weight allocated");

        startingEntropy =
                Math.log(sumOfWeights) -
                        sumOfWeightLogWeights /
                                sumOfWeights;

        sumsOfOnes = new int[FMX * FMY];
        sumsOfWeights = new double[FMX * FMY];
        sumsOfWeightLogWeights = new double[FMX * FMY];
        entropies = new double[FMX * FMY];
        stack = new StackEntry[wave.length * T];
        stacksize = 0;
        System.out.println("Stack allocated");
    }

    Boolean observe() {
        double min = 1e+3;
        int argmin = -1;

        for (int i = 0; i < wave.length; i++) {
            if (onBoundary(i % FMX, i / FMX)) continue;

            int amount = sumsOfOnes[i];
            if (amount == 0) return false;

            double entropy = entropies[i];

            if (amount > 1 && entropy <= min) {
                double noise = 1e-6 * random.nextDouble();
                if (entropy + noise < min) {
                    min = entropy + noise;
                    argmin = i;
                }
            }
        }

        if (argmin == -1) {
            observed = new int[wave.length];
            for (int i = 0; i < wave.length; i++)
                for (int t = 0; t < T; t++)
                    if (wave[i][t]) {
                        observed[i] = t;
                        break;
                    }
            return true;
        }

        double[] distribution = new double[T];
        for (int t = 0; t < T; t++)
            distribution[t] = wave[argmin][t] ? weights[t] : 0;

        int r = Model.randomIndice(distribution, random.nextDouble());

        boolean[] w = wave[argmin];
        for (int t = 0; t < T; t++)
            if (w[t] != (t == r))
                this.ban(argmin, t);

        return null;
    }

    protected void ban(int i, int t) {
        wave[i][t] = false;

        int[] comp = compatible[i][t];
        for (int d = 0; d < 4; d++)
            comp[d] = 0;
        stack[stacksize] = new StackEntry(i, t);
        stacksize++;

        sumsOfOnes[i] -= 1;
        sumsOfWeights[i] -= weights[t];
        sumsOfWeightLogWeights[i] -= weightLogWeights[t];

        double sum = sumsOfWeights[i];
        entropies[i] = Math.log(sum) - sumsOfWeightLogWeights[i] / sum;
    }

    protected void propagate() {
        while (this.stacksize > 0) {
            StackEntry e1 = this.stack[this.stacksize - 1];
            this.stacksize--;

            int i1 = e1.getFirst();
            int x1 = i1 % this.FMX;
            int y1 = i1 / this.FMX;

            for (int d = 0; d < 4; d++) {
                int dx = Model.DX[d], dy = Model.DY[d];
                int x2 = x1 + dx, y2 = y1 + dy;

                if (this.onBoundary(x2, y2)) continue;

                if (x2 < 0) x2 += this.FMX;
                else if (x2 >= this.FMX) x2 -= this.FMX;
                if (y2 < 0) y2 += this.FMY;
                else if (y2 >= this.FMY) y2 -= this.FMY;

                int i2 = x2 + y2 * this.FMX;
                int[] p = this.propagator[d][e1.getSecond()];
                int[][] compat = this.compatible[i2];

                for (int l = 0; l < p.length; l++) {
                    int t2 = p[l];
                    int[] comp = compat[t2];

                    comp[d]--;

                    if (comp[d] == 0) this.ban(i2, t2);
                }
            }
        }
    }

    public boolean run(int seed, int limit) {
        System.out.println("Initializing: ");
        if (wave == null)
            init();

        Clear();
        random = new Random(seed);

        for (int l = 0; l < limit || limit == 0; l++) {
            Boolean result = observe();
            if (result != null)
                return result;
            System.out.println("Observed: " + l);
            propagate();
            System.out.println("Propagated: " + l);
        }

        return true;
    }

    protected void Clear() {
        for (int i = 0; i < this.wave.length; i++) {
            for (int t = 0; t < this.T; t++) {
                this.wave[i][t] = true;
                for (int d = 0; d < 4; d++)
                    this.compatible[i][t][d] =
                            this.propagator[Model.oppposite[d]][t].length;
            }

            this.sumsOfOnes[i] = this.weights.length;
            this.sumsOfWeights[i] = this.sumOfWeights;
            this.sumsOfWeightLogWeights[i] = this.sumOfWeightLogWeights;
            this.entropies[i] = this.startingEntropy;
        }
    }
}
