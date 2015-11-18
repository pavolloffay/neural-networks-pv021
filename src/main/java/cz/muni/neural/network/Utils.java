package cz.muni.neural.network;

import java.util.Arrays;
import java.util.List;

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
}
