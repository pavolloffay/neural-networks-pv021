package cz.muni.neural.network;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

import org.junit.Test;

import cz.muni.neural.network.linear.algebra.DoubleMatrix;

/**
 * @author Pavol Loffay
 */
public class DoubleMatrixTest {

    @Test
    public void testSigmoidZero() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(0, 3, 4);

        doubleMatrix = doubleMatrix.applyOnEach(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByIndex(0,0), is(equalTo(0.5D)));
    }

    @Test
    public void testSigmoidInf() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1000, 3, 4);

        doubleMatrix = doubleMatrix.applyOnEach(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByIndex(0,0), is(equalTo(1D)));
    }

    @Test
    public void testSigmoidMinusInf() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(-1000, 3, 4);

        doubleMatrix = doubleMatrix.applyOnEach(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByIndex(0,0), is(equalTo(0D)));
    }

    @Test
    public void testTranspose1x1() {
        double[][] x = new double[][]{{1}};
        DoubleMatrix doubleMatrix = new DoubleMatrix(x);
        doubleMatrix = doubleMatrix.transpose();

        double[][] expected = new double[][]{{1}};
        assertThat(doubleMatrix.isTheSameAs(new DoubleMatrix(expected)), is(Boolean.TRUE));
    }

    @Test
    public void testTranspose3x3() {
        double[][] x = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        DoubleMatrix doubleMatrix = new DoubleMatrix(x);
        doubleMatrix = doubleMatrix.transpose();

        double[][] expected = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        assertThat(doubleMatrix.isTheSameAs(new DoubleMatrix(expected)), is(Boolean.TRUE));

    }

    @Test
    public void testTranspose3x2() {
        double[][] x = new double[][]{{1, 2}, {3, 4}, {5, 6}};
        DoubleMatrix doubleMatrix = new DoubleMatrix(x);
        doubleMatrix = doubleMatrix.transpose();

        double[][] expected = new double[][]{{1, 3, 5}, {2, 4, 6}};
        assertThat(doubleMatrix.isTheSameAs(new DoubleMatrix(expected)), is(Boolean.TRUE));
    }
    // todo test transpose 2x3

    @Test
    public void testMultiply() {
        double[][] aData = new double[][]{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
        DoubleMatrix a = new DoubleMatrix(aData);

        double[][] bData = new double[][]{{2}, {2}, {2}};
        DoubleMatrix b = new DoubleMatrix(bData);

        DoubleMatrix y = a.matrixMultiply(b);

        double[][] expected = new double[][]{{6}, {12}, {18}};
        assertThat(y.isTheSameAs(new DoubleMatrix(expected)), is(Boolean.TRUE));
    }
}
