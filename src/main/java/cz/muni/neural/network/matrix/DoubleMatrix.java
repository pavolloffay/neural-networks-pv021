package cz.muni.neural.network.matrix;

import static java.lang.Math.abs;

import java.util.Arrays;
import java.util.function.Function;

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

        this.numberOfRows = x.length;
        this.numberOfColumns = x[0].length;
        data = new double[numberOfRows][numberOfColumns];

        for (int row = 0; row < numberOfRows; row++) {
            data[row] = Arrays.copyOf(x[row], numberOfColumns);
        }
    }

    public DoubleMatrix(double value, int numberOfRows, int numberOfColumns) {
        this.data = new double[numberOfRows][numberOfColumns];
        this.numberOfRows = numberOfRows;
        this.numberOfColumns = numberOfColumns;

        for (int row = 0; row < numberOfRows; row++) {
            Arrays.fill(this.data[row], value);
        }
    }

    public int getNumberOfRows() {
        return numberOfRows;
    }

    public int getNumberOfColumns() {
        return numberOfColumns;
    }

    public double getByIndex(int row, int col) {
        return this.data[row][col];
    }

    public void setByIndex(double value, int row, int col) {
        this.data[row][col] = value;
    }
    
    public double[] getRow(int row) {
        return data[row];
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

        double result[][] = new double[this.numberOfRows][that.getNumberOfColumns()];
        for (int row = 0; row < this.numberOfRows; row++) {
            for (int col = 0; col < that.getNumberOfColumns(); col++) {

                double sum = 0;
                for (int i = 0; i < this.numberOfColumns; i++) {
                    sum += this.data[row][i] * that.getByIndex(i, col);
                }

                result[row][col] = sum;
            }
        }

       return new DoubleMatrix(result);
    }

    public DoubleMatrix multiplyByElements(DoubleMatrix that) {
        if (this.numberOfRows != that.getNumberOfRows() ||
                this.numberOfColumns != that.getNumberOfColumns()) {
            throw new IllegalArgumentException("Matrix size does not match for multipleByElements");
        }

        double[][] result = new double[this.numberOfRows][this.getNumberOfColumns()];

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] * that.getByIndex(row, col);
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix subtract(DoubleMatrix that) {
        if (this.numberOfRows != that.getNumberOfRows() ||
                this.numberOfColumns != that.getNumberOfColumns()) {
            throw new IllegalArgumentException("Matrix size does not match for subtraction");
        }

        double[][] result = new double[numberOfRows][numberOfColumns];
        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] - that.getByIndex(row, col);
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix scalarSubstract(double scalar) {
        double[][] result = new double[numberOfRows][numberOfColumns];
        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] - scalar;
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix sum(DoubleMatrix that) {
        if (this.numberOfRows != that.getNumberOfRows() ||
                this.numberOfColumns != that.getNumberOfColumns()) {
            throw new IllegalArgumentException("Matrix size does not match for subtraction");
        }

        double[][] result = new double[numberOfRows][numberOfColumns];
        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row][col] + that.getByIndex(row, col);
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix addFirstColumn(double value) {

        int newNumberOfColumns = this.numberOfColumns + 1;
        double[][] result = new double[numberOfRows][newNumberOfColumns];

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < newNumberOfColumns; col++) {

                result[row][col] = col == 0 ? value : this.data[row][col - 1];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix addFirstRow(double value) {
        int newNumberOfRows = this.numberOfRows + 1;
        double[][] result = new double[newNumberOfRows][numberOfColumns];

        for (int row = 0; row < newNumberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {

                result[row][col] = row == 0 ? value : this.data[row - 1][col];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix removeFirstRow() {
        if (this.numberOfRows == 1) {
            throw new IllegalArgumentException("Matrix is too small");
        }

        int newNumberOfRows = this.numberOfRows - 1;
        double[][] result = new double[newNumberOfRows][numberOfColumns];

        for (int row = 0; row < newNumberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = this.data[row + 1][col];
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix removeFirstColumn() {
        if (this.numberOfColumns == 1) {
            throw new IllegalArgumentException("Matrix is too small");
        }

        int newNumberOfColumns = this.numberOfColumns - 1;
        double[][] result = new double[numberOfRows][newNumberOfColumns];

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < newNumberOfColumns; col++) {
                result[row][col] = this.data[row][col + 1];
            }
        }

        return new DoubleMatrix(result);
    }

    public double[] maxValueInRow() {

        double[] result = new double[numberOfRows];

        for (int row = 0; row < numberOfRows; row++) {
            double max = Double.MIN_VALUE;
            for (int col = 0; col < numberOfColumns; col++) {
                if (max < this.data[row][col]) {
                    max = this.data[row][col];
                }
            }
            result[row] = max;
        }

        return result;
    }

    public DoubleMatrix applyOnEach(Function<Double, Double> fce) {

        double[][] result = new double[numberOfRows][numberOfColumns];

        for(int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                result[row][col] = fce.apply(this.data[row][col]);
            }
        }

        return new DoubleMatrix(result);
    }

    public DoubleMatrix copy() {
        return new DoubleMatrix(this.data);
    }

    public boolean isTheSameAs(DoubleMatrix that) {
        if (this.numberOfRows != that.getNumberOfRows() ||
                this.getNumberOfColumns() != that.getNumberOfColumns()) {
            return false;
        }

        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                if (data[row][col] != that.getByIndex(row, col)) {
                    return false;
                }
            }
        }

        return true;
    }

    public void printSize() {
        System.out.println("Size of matrix = " + numberOfRows + "x" + numberOfColumns);
    }

    @Override
    public String toString() {
        return "DoubleMatrix{" +
                "numberOfColumns=" + numberOfColumns +
                ", numberOfRows=" + numberOfRows +
                ", data=" + Arrays.deepToString(data) +
                '}';
    }

    public boolean allZero() {
        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                if (abs(this.data[row][col]) != 0) {
                    return false;
                }
            }
        }

        return true;
    }
}
