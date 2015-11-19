package cz.muni.neural.network;

import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Pavol Loffay
 */
public class TestUtils {

    public static final String IMAGES_TEST = "t10k-images-idx3-ubyte";
    public static final String LABELS_TEST = "t10k-labels-idx1-ubyte";

    public static final String IMAGES_TRAIN = "train-images-idx3-ubyte";
    public static final String LABELS_TRAIN = "train-labels-idx1-ubyte";

    public static final String IMAGES_TEST_PATH;
    public static final String LABELS_TEST_PATH;
    public static final String IMAGES_TRAIN_PATH;
    public static final String LABELS_TRAIN_PATH;
    static {
        ClassLoader classLoader = TestUtils.class.getClassLoader();

        URL testFile = classLoader.getResource(IMAGES_TEST);
        IMAGES_TEST_PATH = testFile.getPath();
        testFile = classLoader.getResource(LABELS_TEST);
        LABELS_TEST_PATH = testFile.getPath();
        testFile = classLoader.getResource(LABELS_TRAIN);
        LABELS_TRAIN_PATH = testFile.getPath();
        testFile = classLoader.getResource(IMAGES_TRAIN);
        IMAGES_TRAIN_PATH = testFile.getPath();
    }

    public static List<Layer> create3Layers(int features, int[] structure) {
        List<Layer> layers = new ArrayList<>();
        Layer layer1 = new Layer(features);
        Layer layer2 = new Layer(structure[0]);
        Layer layer3 = new Layer(structure[1]);
        layers.add(layer1);
        layers.add(layer2);
        layers.add(layer3);

        return layers;
    }

    public static List<LabeledPoint> createPoints(int number, int features) {

        List<LabeledPoint> labeledPoints = new ArrayList<>();

        Random rand = new Random();
        double[] x = new double[features];
        for (int i = 0; i < features; i++) {
            x[i] = rand.nextDouble() + 5;
        }

        for (int i = 0; i < number; i++) {
            LabeledPoint labeledPoint = new LabeledPoint(5, Arrays.copyOf(x, features));
            labeledPoints.add(labeledPoint);
        }

        return labeledPoints;
    }

    public static int indexOfMaxValue(double[] arr) {

        double max = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < arr.length; i++) {
            if (max < arr[i]) {
                index = i;
                max = arr[i];
            }
        }

        return index;
    }

}
