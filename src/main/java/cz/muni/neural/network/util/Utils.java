package cz.muni.neural.network.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import cz.muni.neural.network.Functions;
import cz.muni.neural.network.matrix.DoubleMatrix;
import cz.muni.neural.network.model.LabeledPoint;

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
    
    public static double mae(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array sizes do not match!");
        }
            
        double sum = 0;
        int count = 0;
        for (int i = 0; i < a.length; i++)
        {
            sum += Math.abs((double)(a[i] - b[i]));
            count++;  
        }
        return sum/count;
    }
    
    public static double mean(List<LabeledPoint> labeledPoints) {
        double firstColSum = 0D;
        if (labeledPoints.isEmpty() || labeledPoints.get(0).getFeatures().length == 0) {
            throw new IllegalArgumentException("List or features are empty!");
        }
        
        int features = labeledPoints.get(0).getFeatures().length;
        
        for (LabeledPoint lp : labeledPoints) {
            if (lp.getFeatures().length != features) {
                throw new IllegalArgumentException("Number of features does not match!");
            }
            firstColSum += lp.getFeatures()[0];
        }
        
            
        double mean = firstColSum / labeledPoints.size();
        return mean;
    }
    
    public static double deviation(List<LabeledPoint> labeledPoints, double mean) {
        double varSum = 0;
        for (LabeledPoint lp : labeledPoints) {
            varSum += Math.pow((lp.getFeatures()[0] - mean), 2);
        }

        double variance = varSum / labeledPoints.size();
        double deviation = Math.sqrt(variance);
        return deviation;
    }
    
    public static List<LabeledPoint> sigmoidNormalize(List<LabeledPoint> labeledPoints, double mean, double deviation) {
        Function sigm = new Functions.Sigmoid();
        
        double inverseDev = 1D / deviation;

        List<LabeledPoint> labeledPointsNormalized = new ArrayList<>();
        for (LabeledPoint lp : labeledPoints) {
            double[] normalizedFeatures = new double[lp.getFeatures().length];
            for (int i = 0; i < lp.getFeatures().length; i++) {
                normalizedFeatures[i] = (double)sigm.apply((lp.getFeatures()[i] - mean)*inverseDev);
            }
            double normalizedLabel = (double)sigm.apply((lp.getLabel() - mean)*inverseDev);
            labeledPointsNormalized.add(new LabeledPoint(normalizedLabel, normalizedFeatures));
        }
        return labeledPointsNormalized;
    }
    
    public static double sigmoidDenormalize(double value, double mean, double deviation) {
        Function logit = new Functions.Logit();

        //System.out.println("Denormalizing. Mean: "+mean+", deviation: "+deviation);
        
        double result = (double)logit.apply(value);
        result = result * deviation;
        result = result + mean;
        
        return result;
    }

}
