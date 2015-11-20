package cz.muni.neural.network;

import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.reader.MNISTReader;

/**
 * @author Pavol Loffay
 */
public class NeuralNetworkDummyTest {


    @Test
    public void testNetworkCreation() throws Exception {

        List<LabeledPoint> labeledPointList = MNISTReader.read(TestUtils.IMAGES_TEST_PATH,
                TestUtils.LABELS_TEST_PATH, 50);
        int features = labeledPointList.get(0).getFeatures().length;

        List<Layer> layers = TestUtils.create3Layers(features, new int[]{25, 10});
        NeuralNetwork network = new NeuralNetwork(layers, 0.05, 50, true,1.5);
    }

    @Test
    public void testWithDummyData() {
        List<LabeledPoint> labeledPoints = TestUtils.createPoints(50, 15);
        List<Layer> layers = TestUtils.create3Layers(labeledPoints.get(0).getFeatures().length, new int[]{25, 10});

        NeuralNetwork network = new NeuralNetwork(layers, 0.05, 50, true, 1.5);
        network.train(labeledPoints);
    }
}
