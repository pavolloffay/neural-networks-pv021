package cz.muni.neural.network;

import java.util.List;

import cz.muni.neural.network.linear.algebra.DoubleMatrix;

/**
 * @author Pavol Loffay
 */
public class NeuralNetwork {

    private final List<Layer> layers;
    private final int numOfLayers;

    private DoubleMatrix[] theta;

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
        this.numOfLayers = layers.size();

        theta = new DoubleMatrix[numOfLayers - 1];

    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void train(List<LabeledPoint> labeledPoints) {

        for (LabeledPoint labeledPoint: labeledPoints) {
            if (labeledPoint.getFeatures().length != layers.get(0).getNumberOfUnits()) {
                throw new IllegalArgumentException("Labeled point has wrong number of features");
            }


            DoubleMatrix a1 = Utils.labeledPointToDoubleMatrix(labeledPoint);
            a1 = a1.transpose();
            // add bias
            a1 = a1.addFristRow(1);


            a1.computeValues(new Utils.Sigmoid());
        }
    }

    public void predict(LabeledPoint labeledPoint) {
        // TODO
    }
}
