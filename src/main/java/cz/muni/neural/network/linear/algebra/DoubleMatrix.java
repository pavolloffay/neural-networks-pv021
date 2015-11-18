package cz.muni.neural.network.linear.algebra;

import java.util.Arrays;
import java.util.List;

import cz.muni.neural.network.LabeledPoint;

/**
 * @author Pavol Loffay
 */
public class DoubleMatrix {

    private final double data[][];
    private final int numberOfRows;
    private final int numberOfColumns;

    public DoubleMatrix(double x[][]) {
        if (x == null) {
            throw new IllegalArgumentException("Array is null");
        }

        this.numberOfColumns = x[0].length;
        this.numberOfRows = x[1].length;
        data = new double[numberOfRows][numberOfColumns];

        for (int row = 0; row < x[0].length; row++) {
            Arrays.copyOf(x[row], x[row].length);
        }
    }

    public DoubleMatrix(List<LabeledPoint> points, int numberOfColumns) {
        if (points == null) {
            throw new IllegalArgumentException("Points are null");
        }

        int numberOfRows = points.size();
        double[][] matrix = new double[numberOfRows][numberOfColumns];

        for (int i = 0; i < numberOfRows; i++) {
            LabeledPoint labeledPoint = points.get(i);

            if (labeledPoint.getFeatures().length != numberOfColumns) {
                throw new IllegalArgumentException("Number of features is illegal");
            }

            // create copy of array
            matrix[i] = Arrays.copyOf(labeledPoint.getFeatures(), numberOfColumns);
        }

        this.data = matrix;
        this.numberOfRows = numberOfRows;
        this.numberOfColumns = numberOfColumns;
    }

    public int getNumberOfRows() {
        return numberOfRows;
    }

    public int getNumberOfColumns() {
        return numberOfColumns;
    }

    public double getByPosition(int row, int col) {
        return this.data[row][col];
    }

    public DoubleMatrix transpose() {
        double[][] result = new double[numberOfColumns][numberOfRows];

        for (int i = 0; i < this.numberOfRows; i++) {
            for (int j = 0; j < this.numberOfColumns; j++) {
                result[j][i] = this.data[i][j];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix scalarMultiply(double scalar) {
        double[][] result = new double[numberOfRows][numberOfColumns];

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] * scalar;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix matrixMultiply(DoubleMatrix that) {
        if (numberOfColumns != that.numberOfRows) {
            throw new IllegalArgumentException("Matrix size does not match for multiplication");
        }

        double result[][] = new double[this.getNumberOfRows()][that.getNumberOfColumns()];
        for (int row = 0; row < this.getNumberOfRows(); row++) {
            for (int col = 0; col < that.getNumberOfColumns(); col++) {

                double sum = 0;
                for (int i = 0; i < that.getNumberOfColumns(); i++) {
                    sum += this.data[row][i] * that.getByPosition(i, col);
                }

                result[row][col] = sum;
            }
        }

       return new DoubleMatrix(result);
    }

    public DoubleMatrix multiplyByElemets(DoubleMatrix that) {
        if (this.numberOfRows != that.getNumberOfRows() ||
                this.numberOfColumns != that.getNumberOfColumns()) {
            throw new IllegalArgumentException("Matrix size does not match for multipleByElements");
        }

        double[][] result = new double[this.numberOfRows][this.getNumberOfColumns()];

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] * that.getByPosition(row, col);
            }
        }

        return new DoubleMatrix(result);
    }
}
