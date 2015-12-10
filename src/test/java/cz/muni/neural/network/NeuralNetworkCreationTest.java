package cz.muni.neural.network;

import java.util.List;

import org.junit.Test;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.util.MNISTReader;
import cz.muni.neural.network.util.TestUtils;

/**
 * @author Pavol Loffay
 */
public class NeuralNetworkCreationTest {

    @Test
    public void testCreation() throws Exception {

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
    public void testCreationAndSimpleRun() {
        List<LabeledPoint> labeledPoints = TestUtils.createPoints(50, 15);

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(0.05)
                .withGradientIterations(10)
                .withRegularize(true)
                .withRegularizeLambda(50)
                .withHypothesisFn(new Functions.Sigmoid())
                .withHypothesisDerivation(new Functions.SigmoidGradient())
                .withInputLayer(labeledPoints.get(0).getFeatures().length)
                .addLayer(4)
                .addLayer(5)
                .addLastLayer(5);

        network.train(labeledPoints);
    }
}
