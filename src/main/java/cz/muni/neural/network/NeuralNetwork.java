package cz.muni.neural.network;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import cz.muni.neural.network.linear.algebra.DoubleMatrix;

/**
 * @author Pavol Loffay
 */
public class NeuralNetwork {
    private static final double EPSILON_INIT_THETA = 0.12D;
    private static final double BIAS = 1D;

    private Function<Double, Double> hypothesis = new Functions.Sigmoid();
    private Function<Double, Double> hypothesisDer = new Functions.SigmoidGradient();

    private final List<Layer> layers;
    private final int numOfLayers;

    private final double gradientAlpha;
    private final long gradientNumberOfIter;
    private final double lambdaRegul;

    private List<DoubleMatrix> thetas;

    private List<LabeledPoint> labeledPoints;

    public NeuralNetwork(List<Layer> layers, double gradientAlpha, long gradientNumberOfIter, double lambdaRegul) {
        this.layers = layers;
        this.numOfLayers = layers.size();
        this.gradientAlpha = gradientAlpha;
        this.gradientNumberOfIter = gradientNumberOfIter;
        this.lambdaRegul = lambdaRegul;

        this.thetas = createThetas(true);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void train(List<LabeledPoint> labeledPoints) {
        this.labeledPoints = labeledPoints;
        gradientDescent();
    }


    /**
     * returns the probability
     */
    @SuppressWarnings("Duplicates")
    public double[] predict(LabeledPoint labeledPoint) {

        List<DoubleMatrix> activations = new ArrayList<>(numOfLayers -1);
        List<DoubleMatrix> zetas = new ArrayList<>(numOfLayers - 1);

        DoubleMatrix a1 = Utils.labeledPointToDoubleMatrix(labeledPoint);
        a1 = a1.transpose();
        a1 = a1.addFirstRow(BIAS);
        activations.add(a1);

        // forward propagation
        for (int layer = 0; layer < numOfLayers - 1; layer++) {

            DoubleMatrix zet = thetas.get(layer).matrixMultiply(activations.get(layer));
            DoubleMatrix activation = zet.applyOnEach(hypothesis);

            if (layer < numOfLayers - 2) {
                // skip last
                activation = activation.addFirstRow(1);
            }

            zetas.add(zet);
            activations.add(activation);
        }

//        activations.get(numOfLayers - 1).printSize();
        return activations.get(numOfLayers - 1).maxValueInRow();
    }

    private void gradientDescent() {

        for (int i = 0; i < gradientNumberOfIter; i++) {

            for (int layer = 0; layer < numOfLayers - 1; layer++) {

                DoubleMatrix theta = thetas.get(layer);
                List<DoubleMatrix> thetasGrad = thetasGrad();

                DoubleMatrix thetaGrad = thetasGrad.get(layer).scalarMultiply(gradientAlpha);
                theta = theta.subtract(thetaGrad);

                thetas.set(layer, theta);
            }

            System.out.println("Gradient iteration = " + i);
        }
    }

    @SuppressWarnings("Duplicates")
    private List<DoubleMatrix> thetasGrad() {

        List<DoubleMatrix> thetasGrad = createThetas(false);
        List<DoubleMatrix> zetas = new ArrayList<>(numOfLayers - 1);
        List<DoubleMatrix> activations = new ArrayList<>(numOfLayers -1);
        List<DoubleMatrix> deltas = new ArrayList<>(numOfLayers - 1);

        for (LabeledPoint labeledPoint: labeledPoints) {
            if (labeledPoint.getFeatures().length != layers.get(0).getNumberOfUnits()) {
                throw new IllegalArgumentException("Labeled point has wrong number of features");
            }

            DoubleMatrix a1 = Utils.labeledPointToDoubleMatrix(labeledPoint);
            a1 = a1.transpose();
            a1 = a1.addFirstRow(BIAS);
            activations.add(a1);

            // forward propagation
            for (int layer = 0; layer < numOfLayers - 1; layer++) {

                DoubleMatrix zet = thetas.get(layer).matrixMultiply(activations.get(layer));
                DoubleMatrix activation = zet.applyOnEach(hypothesis);

                if (layer < numOfLayers - 2) {
                    // skip last
                    activation = activation.addFirstRow(1);
                }

                activations.add(activation);
                zetas.add(zet);
            }


            // back propagation
            DoubleMatrix lastDelta = logicalResultColumnVector(labeledPoint)
                    .subtract(activations.get(numOfLayers - 1));
            deltas.add(lastDelta);
            for (int layer = numOfLayers - 4; layer >= 0; layer++) {

                DoubleMatrix sigmoidGradient = zetas.get(layer).applyOnEach(hypothesisDer);
                sigmoidGradient = sigmoidGradient.addFirstRow(1);

                DoubleMatrix delta = thetas.get(layer).transpose().matrixMultiply(deltas.get(layer + 1));
                delta = delta.multiplyByElements(sigmoidGradient);
                delta = delta.removeFirstRow();

                DoubleMatrix thetaGrad = thetasGrad.get(layer);
                DoubleMatrix thetaGradMul = delta.matrixMultiply(activations.get(layer).transpose());
                thetaGrad = thetaGrad.sum(thetaGradMul);
                thetaGrad = thetaGrad.scalarMultiply(labeledPoints.size());
                thetasGrad.set(layer, thetaGrad);

                // TODO Regularization - Lambda
            }
        }

        double regul = lambdaRegul / labeledPoints.size();
        for (int i = 0; i < thetasGrad.size(); i++) {
            DoubleMatrix thetaGrad = thetasGrad.get(i);

            DoubleMatrix thetaRegul = thetas.get(i).removeFirstColumn().addFirstColumn(0).scalarMultiply(regul);
            thetaGrad = thetaGrad.sum(thetaRegul);

            thetasGrad.set(i, thetaGrad);
        }
        return thetasGrad;
    }

    private DoubleMatrix logicalResultColumnVector(LabeledPoint labeledPoint) {
        int numberOfClasses = layers.get(layers.size() -1).getNumberOfUnits();

        double[][] arr = new double[numberOfClasses][1];
        for (int col = 0; col < numberOfClasses; col++) {
            arr[col][0] = labeledPoint.getLabel() == col ? 1D : 0D;
        }

        return new DoubleMatrix(arr);
    }

    /**
     *
     * @param random, if false thetas are initialized with 0
     * @return
     */
    private List<DoubleMatrix> createThetas(boolean random) {
        List<DoubleMatrix> thetas = new ArrayList<>(numOfLayers - 1);

        for (int layer = 0; layer < numOfLayers - 1; layer++) {

            int rows = layers.get(layer + 1).getNumberOfUnits();
            int cols =   layers.get(layer).getNumberOfUnits() + 1;

            DoubleMatrix theta = random ?
                    Utils.randomMatrix(EPSILON_INIT_THETA, rows, cols) :
                    new DoubleMatrix(0, rows, cols);

            thetas.add(theta);
        }

        return thetas;
    }
}
