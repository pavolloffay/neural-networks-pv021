package cz.muni.neural.network;

import java.util.Collections;
import java.util.List;

import cz.muni.neural.network.linear.algebra.DoubleMatrix;

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


            DoubleMatrix a1 = Utils.labeledPointsToDoubleMatrix(Collections.singletonList(labeledPoint));


        }



    }

    public void predict(LabeledPoint labeledPoint) {
        // TODO
    }
}
