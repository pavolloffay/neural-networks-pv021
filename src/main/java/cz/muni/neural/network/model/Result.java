package cz.muni.neural.network.model;

import java.util.Arrays;

/**
 * @author Pavol Loffay
 */
public class Result {

    private double[] data;
    private int maxIndex;

    public Result(double[] data) {
        this.data = data;

        this.maxIndex = indexOfMaxValue(data);
    }

    public double[] getData() {
        return data;
    }

    public int getMaxIndex() {
        return maxIndex;
    }

    public static int indexOfMaxValue(double[] arr) {

        double max = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < arr.length; i++) {
            if (max < arr[i]) {
                index = i;
                max = arr[i];
            }
        }

        return index;
    }

    @Override
    public String toString() {
        return "Result{" +
                "maxIndex=" + maxIndex +
                ", data=" + Arrays.toString(data) +
                '}';
    }
}
