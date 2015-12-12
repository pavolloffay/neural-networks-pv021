package cz.muni.neural.network;

import cz.muni.neural.network.model.LabeledPoint;
import cz.muni.neural.network.model.Result;
import cz.muni.neural.network.util.CSVReader;
import cz.muni.neural.network.util.MNISTReader;
import cz.muni.neural.network.util.OHLCReader;
import cz.muni.neural.network.util.Utils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Pavol Loffay
 */
public class App {
    
    private static int TRAIN = 500;
    private static int TEST = 100; 
    private static int ITER = 200;
    private static double ALPHA = 0.5;
    private static boolean REGULARIZE = true;
    private static double LAMBDA = 1;
    private static int FEATURES = 20;
    private static String OHLC_PATH = "c:\\ohlc\\GBPUSD.csv";
    private static String RESULT_PATH = "c:\\ohlc\\result.csv";
    private static String MNIST_TRAIN_PATH = "c:\\mnist\\train-images-idx3-ubyte";
    private static String MNIST_LABELS_TRAIN_PATH = "c:\\mnist\\train-labels-idx1-ubyte";
    private static String MNIST_TEST_PATH = "c:\\mnist\\t10k-images-idx3-ubyte";
    private static String MNIST_LABELS_TEST_PATH = "c:\\mnist\\t10k-labels-idx1-ubyte";
    private static int PERIOD = 60;
    private static List<Integer> architecture = Arrays.asList(15);
    
    private static NeuralNetwork network = null;    

    public static void main(String[] args) throws IOException {
        System.out.println("Neural network for classification and prediction.");
        System.out.println("Authors: Pavol Loffay, Vaclav Blahut and Jan Brandejs.");
        
        boolean quit = false;
        while (!quit) {
            System.out.println("1. MNIST image classification");
            System.out.println("2. OHLC time series prediction");
            System.out.println("q to quit");            
            System.out.print("Your choice: ");
        
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));        
            String line = br.readLine();
        
            switch (line) {
                case "1":
                    predictMNISTsettings();                    
                    break;
                case "2":
                    predictOHLCsettings();
                    break;
                case "q":
                    quit = true;
                    break;
            }
            
        }
    }
    
    private static void predictOHLCsettings() throws IOException {        
                
        architecture = Arrays.asList(15);
        
        boolean quit = false;
        while (!quit) {
            System.out.println();
            System.out.println("OHLC time series prediction menu");
            System.out.println("Edit settings:");
            System.out.println();            
            printCommonSettings();            
            System.out.println("7. Length of one example: "+FEATURES);
            System.out.println("8. OHLC file path: "+OHLC_PATH);            
            System.out.println("9. result file path: "+RESULT_PATH);            
            System.out.println("10. period (in seconds): "+PERIOD);            
            System.out.println();
            System.out.println("a to view or change network architecture");            
            System.out.println("b to predict using baseline");            
            System.out.println("s to start");            
            System.out.println("q to quit");            
            System.out.print("Your choice: ");
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));        
            String line = br.readLine();
            switch (line) {
                case "1":
                case "2":
                case "3":
                case "4":
                case "5":
                case "6":
                    editCommonSettings(line);
                    break;
                case "7":
                    System.out.print("Enter new length of one example: ");                    
                    line = br.readLine();
                    try {
                        FEATURES = Integer.parseInt(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "8":
                    System.out.print("Enter new path to OHLC file: ");                    
                    OHLC_PATH = br.readLine();                 
                    break;
                case "9":
                    System.out.print("Enter new path to result file: ");                    
                    RESULT_PATH = br.readLine();                 
                    break;
                case "10":
                    System.out.print("Enter new period of data in seconds: ");                    
                    line = br.readLine();
                    try {
                        PERIOD = Integer.parseInt(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "a":
                    editArchitecture();
                    break;
                case "b":
                    predictOHLCbaseline();
                    break;
                case "q":
                    quit = true;
                    break;
                case "s":
                    predictOHLC();
                    break;
            }
            
        }
    }
    
    
    /**
     * Predict just by using last value
     * @throws IOException 
     */    
    private static void predictOHLCbaseline() throws IOException {     
        
        List<LabeledPoint> testPoints = OHLCReader.read(OHLC_PATH, FEATURES, TEST, true, PERIOD);
        
        if (testPoints.size() < TEST) {
            System.out.println("Not enough points in file!");
            return;
        }        
        
        double[] labels = new double[TEST];
        double[] predictions = new double[TEST];
        for (int i = 0; i < TEST; i++) {

            LabeledPoint labeledPoint = testPoints.get(i);
 
            labels[i] = labeledPoint.getLabel();
            predictions[i] = labeledPoint.getFeatures()[FEATURES-1];
        }
        

        Double rmse = Utils.rmse(labels, predictions);
        
        System.out.println("Test examples = " + testPoints.size());
        System.out.printf("RMSE: %.9f", rmse);
        System.out.println();
    }
    
    private static void predictOHLC() throws IOException {     
        
        List<LabeledPoint> allPoints = OHLCReader.read(OHLC_PATH, FEATURES, TRAIN+TEST, true, PERIOD);
        
        if (allPoints.size() < TRAIN + TEST) {
            System.out.println("Not enough points in file!");
            return;
        }        
        
        double mean = Utils.mean(allPoints);
        double deviation = Utils.deviation(allPoints, mean);
        
        List<LabeledPoint> trainPoints = allPoints.subList(0, TRAIN);
        List<LabeledPoint> testPoints = allPoints.subList(TRAIN, TRAIN+TEST);
        
        System.out.println("Normalizing. Mean: "+mean+", Deviation: "+deviation);
        
        List<LabeledPoint> normalizedTrainPoints = Utils.sigmoidNormalize(trainPoints, mean, deviation);
        List<LabeledPoint> normalizedTestPoints = Utils.sigmoidNormalize(testPoints, mean, deviation);

        network = NeuralNetwork.newBuilder()
                .withGradientAlpha(ALPHA)
                .withGradientIterations(ITER)
                .withRegularize(REGULARIZE)
                .withRegularizeLambda(LAMBDA)
                .withInputLayer(FEATURES)
                .addLayers(architecture)
                .addLastLayer(1);

        network.train(normalizedTrainPoints);
        

        double[] labels = new double[normalizedTestPoints.size()];
        double[] predictions = new double[normalizedTestPoints.size()];
        for (int i = 0; i < normalizedTestPoints.size(); i++) {

            LabeledPoint normalizedLabeledPoint = normalizedTestPoints.get(i);
            LabeledPoint labeledPoint = testPoints.get(i);
            Result result = network.predict(normalizedLabeledPoint);

            double prediction = Utils.sigmoidDenormalize(result.getData()[0], mean, deviation);
            
            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + prediction);  
            labels[i] = labeledPoint.getLabel();
            predictions[i] = prediction;
        }

        Double rmse = Utils.rmse(labels, predictions);
        
        try {
            CSVReader.write(RESULT_PATH, ";", labels, predictions);
        } catch (Exception e) {
            System.out.println("Result file writing failed.");
        }

        System.out.println("Test examples = " + normalizedTestPoints.size());
        System.out.printf("RMSE: %.9f", rmse);
        System.out.println();
    }
    
    private static void printCommonSettings() {
        System.out.println("1. Train points max. count: "+TRAIN);
        System.out.println("2. Test points max. count: "+TEST);
        System.out.println("3. Learning iterations: "+ITER);
        System.out.println("4. Learning alpha: "+ALPHA);
        System.out.println("5. Regularization: "+(REGULARIZE ? "ON" : "OFF"));
        System.out.println("6. Regularization lambda: "+LAMBDA);        
    }
    
    private static void editCommonSettings(String option) throws IOException {      
        
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
            String line = "";
            switch (option) {
                case "1":
                    System.out.print("Enter new number of train points: ");                    
                    line = br.readLine();
                    try {
                        TRAIN = Integer.parseInt(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "2":
                    System.out.print("Enter new number of test points: ");                    
                    line = br.readLine();
                    try {
                        TEST = Integer.parseInt(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "3":
                    System.out.print("Enter new number of learning iterations: ");                    
                    line = br.readLine();
                    try {
                        ITER = Integer.parseInt(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "4":
                    System.out.print("Enter new learning alpha (decimal): ");                    
                    line = br.readLine();
                    try {
                        ALPHA = Double.parseDouble(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
                case "5":
                    REGULARIZE = !REGULARIZE;
                    System.out.println("Regularization "+(REGULARIZE ? "ON" : "OFF"));                    
                    break;                
                case "6":
                    System.out.print("Enter new regularization lambda (decimal): ");                    
                    line = br.readLine();
                    try {
                        LAMBDA = Double.parseDouble(line);
                    } catch (NumberFormatException e) {
                        System.out.println("Wrong format.");      
                    }                        
                    break;
            }            
    }
    
    private static void editArchitecture() throws IOException {    
        
        System.out.println();
        System.out.println("Current architecture:");
        for (int i = 0; i < architecture.size(); i++) {
            System.out.println((i+1)+". hidden layer has "+architecture.get(i)+" neurons.");            
        }
        System.out.println();
        System.out.println("Define new architecture? y/n");
        
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
        String line = br.readLine();
        
        if (line.equals("y")) {
            architecture = new ArrayList<Integer>();
            System.out.println();
            System.out.print("Enter new number of hidden layers: ");
            line = br.readLine();
            int layers = 0;
            try {
                layers = Integer.parseInt(line);
            } catch (NumberFormatException e) {
                System.out.println("Wrong format.");      
                return;
            }   
            
            for (int i = 0; i < layers; i++) {
                System.out.println((i+1)+". hidden layer neurons: ");            
                line = br.readLine();
                try {
                    architecture.add(Integer.parseInt(line));
                } catch (NumberFormatException e) {
                    System.out.println("Wrong format.");      
                    return;
                }   
            }
            
            System.out.println("Succesfuly changed.");            
        }
                 
    }
    
    private static void predictMNISTsettings() throws IOException {        
                
        architecture = Arrays.asList(30);
        
        boolean quit = false;
        while (!quit) {
            System.out.println();
            System.out.println("MNIST handwritted numbers classification menu");
            System.out.println("Edit settings:");
            System.out.println();            
            printCommonSettings();            
            System.out.println("7. MNIST train images file: "+MNIST_TRAIN_PATH);
            System.out.println("8. MNIST train labels file: "+MNIST_LABELS_TRAIN_PATH);
            System.out.println("9. MNIST test images file: "+MNIST_TEST_PATH);
            System.out.println("10. MNIST test labels file: "+MNIST_LABELS_TEST_PATH);
            System.out.println();
            System.out.println("a to view or change network architecture");            
            System.out.println("b to predict using baseline");            
            System.out.println("s to start");            
            System.out.println("q to quit");            
            System.out.print("Your choice: ");
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));        
            String line = br.readLine();
            switch (line) {
                case "1":
                case "2":
                case "3":
                case "4":
                case "5":
                case "6":
                    editCommonSettings(line);
                    break;
                case "7":
                    System.out.print("Enter new path to MNIST train images file: ");                    
                    MNIST_TRAIN_PATH = br.readLine();                 
                    break;
                case "8":
                    System.out.print("Enter new path to MNIST train labels file: ");                    
                    MNIST_LABELS_TRAIN_PATH = br.readLine();                 
                    break;
                case "9":
                    System.out.print("Enter new path to MNIST test images file: ");                    
                    MNIST_TEST_PATH = br.readLine();                 
                    break;
                case "10":
                    System.out.print("Enter new path to MNIST test labels file: ");                    
                    MNIST_LABELS_TEST_PATH = br.readLine();                 
                    break;
                case "a":
                    editArchitecture();
                    break;
                case "q":
                    quit = true;
                    break;
                case "s":
                    predictMNIST();
                    break;
            }
            
        }
    }
    
    public static void predictMNIST() throws IOException {

        List<LabeledPoint> trainPoints = MNISTReader.read(MNIST_TRAIN_PATH,
                MNIST_LABELS_TRAIN_PATH, TRAIN);
        int features = trainPoints.get(0).getFeatures().length;

        NeuralNetwork network = NeuralNetwork.newBuilder()
                .withGradientAlpha(ALPHA)
                .withGradientIterations(ITER)
                .withRegularize(REGULARIZE)
                .withRegularizeLambda(LAMBDA)
                .withInputLayer(features)
                .addLayers(architecture)
                .addLastLayer(10);

        /**
         * train
         */
        network.train(trainPoints);

        /**
         * test
         */
        List<LabeledPoint> testPoints = MNISTReader.read(MNIST_TEST_PATH,
                MNIST_LABELS_TEST_PATH, TEST);

        int ok = 0;
        for (LabeledPoint labeledPoint: testPoints) {

            Result result = network.predict(labeledPoint);

            System.out.println(result);
            System.out.println("Label = " + labeledPoint.getLabel() + " predicted = " + result.getMaxIndex());
            if (labeledPoint.getLabel() == result.getMaxIndex()) {
                ok++;
            }
        }

        Double success = (ok / (double)testPoints.size()) * 100D;

        System.out.println("\n\nSuccessfully predicted = " + ok);
        System.out.println("Test examples = " + testPoints.size());
        System.out.println("Success = " + success + "%");
    }
    
}
