package cz.muni.neural.network;

import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import cz.muni.neural.network.model.LabeledPoint;

/**
 * @author Pavol Loffay
 */
public class TestUtils {

    public static final String IMAGES_TEST = "t10k-images-idx3-ubyte";
    public static final String LABELS_TEST = "t10k-labels-idx1-ubyte";

    public static final String IMAGES_TRAIN = "train-images-idx3-ubyte";
    public static final String LABELS_TRAIN = "train-labels-idx1-ubyte";

    public static final String CSV_TEST = "GBPUSD_M5_test.csv";
    public static final String CSV_TRAIN = "GBPUSD_M5_train.csv";
    
    public static final String CSV_RESULTS = "csv_results.csv";
    
    public static final String IMAGES_TEST_PATH;
    public static final String LABELS_TEST_PATH;
    public static final String IMAGES_TRAIN_PATH;
    public static final String LABELS_TRAIN_PATH;
    public static final String CSV_TEST_PATH;    
    public static final String CSV_TRAIN_PATH;  
    public static final String CSV_RESULTS_PATH;    
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
        testFile = classLoader.getResource(CSV_TEST);
        CSV_TEST_PATH = testFile.getPath();
        testFile = classLoader.getResource(CSV_TRAIN);
        CSV_TRAIN_PATH = testFile.getPath();
        testFile = classLoader.getResource(CSV_RESULTS);
        CSV_RESULTS_PATH = testFile.getPath();
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
}
