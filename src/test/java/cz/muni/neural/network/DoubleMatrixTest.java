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

        doubleMatrix.computeValues(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByPosition(0,0), is(equalTo(0.5D)));
    }

    @Test
    public void testSigmoidInf() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(1000, 3, 4);

        doubleMatrix.computeValues(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByPosition(0,0), is(equalTo(1D)));
    }

    @Test
    public void testSigmoidMinusInf() {
        DoubleMatrix doubleMatrix = new DoubleMatrix(-1000, 3, 4);

        doubleMatrix.computeValues(new Utils.Sigmoid());

        assertThat(doubleMatrix.getByPosition(0,0), is(equalTo(0D)));
    }
}
