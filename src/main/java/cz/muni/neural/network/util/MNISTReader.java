package cz.muni.neural.network.util;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import cz.muni.neural.network.model.LabeledPoint;

/**
 * Source code from
 * @see <a href="https://code.google.com/p/pen-ui/source/browse/trunk/skrui/src/org/six11/skrui/charrec/MNISTReader.java?r=185"></a>
 */
public class MNISTReader {


    public static List<LabeledPoint> read(String imagesFile, String labelsFile, int numberOfPoints) throws IOException {
        System.out.format("MNIST reader: numberOfPoints: %d, file: %s\n", numberOfPoints, imagesFile);

        DataInputStream labels = new DataInputStream(new FileInputStream(labelsFile));
        DataInputStream images = new DataInputStream(new FileInputStream(imagesFile));
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }
        int numLabels = labels.readInt();
        int numImages = images.readInt();
        int numRows = images.readInt();
        int numCols = images.readInt();
        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
            System.exit(0);
        }

        System.out.println("Size of image = " + numRows + "x" + numCols);
        List<LabeledPoint> labeledPoints = new ArrayList<>(numberOfPoints);

        long start = System.currentTimeMillis();
        int numLabelsRead = 0;
        int numImagesRead = 0;
        while (labels.available() > 0 && numLabelsRead < numLabels && numberOfPoints-- > 0) {

            byte label = labels.readByte();
            numLabelsRead++;
            int[][] image = new int[numCols][numRows];
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    image[colIdx][rowIdx] = images.readUnsignedByte();
                }
            }

            numImagesRead++;
            LabeledPoint labeledPoint = new LabeledPoint(label, imageToArray(image));
            labeledPoints.add(labeledPoint);

            // At this point, 'label' and 'image' agree and you can do whatever you like with them.

            if (numLabelsRead % 10 == 0) {
                System.out.print(".");
            }
            if ((numLabelsRead % 800) == 0) {
                System.out.print(" " + numLabelsRead + " / " + numLabels);
                long end = System.currentTimeMillis();
                long elapsed = end - start;
                long minutes = elapsed / (1000 * 60);
                long seconds = (elapsed / 1000) - (minutes * 60);
                System.out.println("  " + minutes + " m " + seconds + " s ");
            }
        }
        System.out.println();
        long end = System.currentTimeMillis();
        long elapsed = end - start;
        long minutes = elapsed / (1000 * 60);
        long seconds = (elapsed / 1000) - (minutes * 60);
        System.out
                .println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");

        return labeledPoints;
    }

    private static double[] imageToArray(int[][] image) {
        double[] result = new double[image.length * image[0].length];

        int columns = image[0].length;

        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[0].length; j++) {
                result[i*columns + j] = image[i][j];
            }
        }

        return result;
    }
}
