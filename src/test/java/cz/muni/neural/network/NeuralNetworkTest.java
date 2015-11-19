package cz.muni.neural.network;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.reader.MNISTReader;

/**
 * @author Pavol Loffay
 */
public class NeuralNetworkTest {

    @Test
    public void testOnImages() throws IOException {

        int TRAIN = 700;
        int TEST = 30;
        double ALPHA = 0.05;
        int ITER = 100;
        double LAMBDA = 1100;

        List<LabeledPoint> trainPoints = MNISTReader.read(TestUtils.IMAGES_TRAIN_PATH,
                TestUtils.LABELS_TRAIN_PATH, TRAIN);
        int features = trainPoints.get(0).getFeatures().length;

        /**
         * train
         */
        List<Layer> layers = TestUtils.create3Layers(features, new int[]{25, 10});
        NeuralNetwork network = new NeuralNetwork(layers, ALPHA, ITER, LAMBDA);
        network.train(trainPoints);

        /**
         * test
         */
        List<LabeledPoint> testPoints = MNISTReader.read(TestUtils.IMAGES_TRAIN_PATH,
                TestUtils.LABELS_TRAIN_PATH, TEST);

        int ok = 0;
        for (LabeledPoint labeledPoint: testPoints) {
            double[] result = network.predict(labeledPoint);

            int index = TestUtils.indexOfMaxValue(result);

            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + index);
            if (labeledPoint.getLabel() == index) {
                ok++;
            }
        }

        System.out.println("\n\nSuccessfully predicted = " + ok);
        System.out.println("Test examples = " + testPoints.size());
    }
}
