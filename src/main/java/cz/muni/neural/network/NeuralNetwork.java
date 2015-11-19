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

    private Function<Double, Double> hypothesis = new Functions.Sigmoid();
    private Function<Double, Double> hypothesisDer = new Functions.SigmoidGradient();

    private final List<Layer> layers;
    private final int numOfLayers;

    private List<DoubleMatrix> thetas;


    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
        this.numOfLayers = layers.size();

        thetas = createRandomThetas();

    }

    public List<Layer> getLayers() {
        return layers;
    }


    public void train(List<LabeledPoint> labeledPoints) {

        List<DoubleMatrix> zetas = new ArrayList<>(numOfLayers - 1);
        List<DoubleMatrix> activations = new ArrayList<>(numOfLayers -1);
        List<DoubleMatrix> deltas = new ArrayList<>(numOfLayers - 1);
        List<DoubleMatrix> thetasGrad = new ArrayList<>(numOfLayers -1);

        for (LabeledPoint labeledPoint: labeledPoints) {
            if (labeledPoint.getFeatures().length != layers.get(0).getNumberOfUnits()) {
                throw new IllegalArgumentException("Labeled point has wrong number of features");
            }

            // forward propagation
            for (int layer = 0; layer < numOfLayers - 1; layer++) {
                if (layer == 1) {
                    DoubleMatrix a1 = Utils.labeledPointToDoubleMatrix(labeledPoint);
                    activations.add(a1);
                    continue;
                }

                DoubleMatrix zLayer = thetas.get(layer).matrixMultiply(activations.get(layer - 1));
                DoubleMatrix aLayer = zLayer.applyOnEach(hypothesis);
                aLayer = aLayer.addFirstRow(1); //add bias todo do not do it for last

                activations.add(aLayer);
                zetas.add(zLayer);
            }
            // back propagation
            for (int layer = numOfLayers - 1; layer >= 0; layer++) {
                if (layer == numOfLayers - 1) {
                    DoubleMatrix lastDelta = null;
                    deltas.add(lastDelta);
                }

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
    }

    public void predict(LabeledPoint labeledPoint) {
        // TODO
    }

    private List<DoubleMatrix> createRandomThetas() {
        List<DoubleMatrix> thetas = new ArrayList<>(numOfLayers - 1);

        for (int layer = 0; layer < numOfLayers - 1; layer++) {

            DoubleMatrix theta = Utils.randomMatrix(EPSILON_INIT_THETA,
                    layers.get(layer + 1).getNumberOfUnits(), layers.get(layer).getNumberOfUnits());

            thetas.add(theta);
        }

        return thetas;
    }
}
