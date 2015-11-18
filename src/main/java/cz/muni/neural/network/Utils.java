package cz.muni.neural.network;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import cz.muni.neural.network.linear.algebra.DoubleMatrix;

/**
 * @author Pavol Loffay
 */
public class Utils {

    public static DoubleMatrix labeledPointsToDoubleMatrix(List<LabeledPoint> labeledPoints) {
        if (labeledPoints == null) {
            throw new IllegalArgumentException("Labeled points were null");
        }

        int numberOfRows = labeledPoints.size();
        int numberOfColumns = labeledPoints.size() > 0 ? labeledPoints.get(0).getFeatures().length : 0;

        double[][] result = new double[numberOfRows][numberOfColumns];

        for (int i = 0; i < numberOfRows; i++) {
            LabeledPoint labeledPoint = labeledPoints.get(i);

            if (labeledPoint.getFeatures().length != numberOfColumns) {
                throw new IllegalArgumentException("Number of features is not same across all features");
            }

            // create copy of array
            result[i] = Arrays.copyOf(labeledPoint.getFeatures(), numberOfColumns);

        }

        return new DoubleMatrix(result);
    }

    public static DoubleMatrix labeledPointToDoubleMatrix(LabeledPoint point) {
        if (point == null) {
            throw new IllegalArgumentException("Labeled points were null");
        }

        return labeledPointsToDoubleMatrix(Arrays.asList(point));
    }

    public static DoubleMatrix randomMatrix(double epsilon, int numberOfRows, int numberOfColumns) {
        double[][] result = new double[numberOfRows][numberOfColumns];

        Random rand = new Random();

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = rand.nextDouble() * 2 * epsilon - epsilon;
            }
        }

        return new DoubleMatrix(result);
    }

    public static class Sigmoid implements Function<Double, Double> {

        public Double apply(Double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    }

    public static class SigmoidGradient implements Function<Double, Double> {

        @Override
        public Double apply(Double x) {
            return (1.0 / (1.0 + Math.exp(-x))) * (1 - (1.0 / (1.0 + Math.exp(-x))));
        }
    }

}
