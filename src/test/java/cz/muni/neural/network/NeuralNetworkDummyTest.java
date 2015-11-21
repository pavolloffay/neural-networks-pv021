package cz.muni.neural.network;

import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.util.MNISTReader;

/**
 * @author Pavol Loffay
 */
public class NeuralNetworkDummyTest {


    @Test
    public void testNetworkCreation() throws Exception {

        List<LabeledPoint> labeledPointList = MNISTReader.read(TestUtils.IMAGES_TEST_PATH,
                TestUtils.LABELS_TEST_PATH, 50);
        int features = labeledPointList.get(0).getFeatures().length;

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(0.05)
                .withGradientIterations(50)
                .withRegularize(true)
                .withRegularizeLambda(50)
                .withInputLayer(features)
                .addLastLayer(5);
    }

    @Test
    public void testWithDummyData() {
        List<LabeledPoint> labeledPoints = TestUtils.createPoints(50, 15);

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(0.05)
                .withGradientIterations(50)
                .withRegularize(true)
                .withRegularizeLambda(50)
                .withInputLayer(labeledPoints.get(0).getFeatures().length)
                .addLayer(4)
                .addLayer(5)
                .addLastLayer(5);

        network.train(labeledPoints);
    }
}
