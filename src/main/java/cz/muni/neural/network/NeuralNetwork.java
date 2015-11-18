package cz.muni.neural.network;

import java.util.List;

/**
 * @author Pavol Loffay
 */
public class NeuralNetwork {

    private final List<Layer> layers;
    private final int numOfLayers;

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
        this.numOfLayers = layers.size();
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void train(List<LabeledPoint> labeledPoints) {

        for (LabeledPoint labeledPoint: labeledPoints) {
            if (labeledPoint.getFeatures().length != layers.get(0).getNumberOfUnits()) {
                throw new IllegalArgumentException("Labeled point has wrong number of features");
            }
        }
    }

    public void predict(LabeledPoint labeledPoint) {
        // TODO
    }
}
