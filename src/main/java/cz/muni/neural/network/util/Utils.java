package cz.muni.neural.network.util;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.matrix.DoubleMatrix;

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
    
    public static double rmse(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array sizes do not match!");
        }
            
        double sum = 0;
        int count = 0;
        for (int i = 0; i < a.length; i++)
        {
            sum += Math.pow((double)(a[i] - b[i]), 2);
            count++;  
        }
        return Math.sqrt(sum/count);
    }

}
