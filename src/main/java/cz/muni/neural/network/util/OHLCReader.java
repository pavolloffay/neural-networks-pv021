/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.neural.network.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.joda.time.DateTime;
import org.joda.time.Seconds;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import cz.muni.neural.network.model.LabeledPoint;

/**
 * @author Václav
 */
public class OHLCReader {

    public static List<LabeledPoint> read(String file, int numberOfFeatures, int numberOfPoints, boolean skipWeekends,
                                          int period) throws IOException {

        BufferedReader br = null;
        String line = "";
        List<Double> values = new ArrayList<Double>();
        List<Integer> weekendIndices = new ArrayList<Integer>();
        DateTime prevDate = null;
        DateTimeFormatter dtf = DateTimeFormat.forPattern("yyyy.MM.dd HH:mm");

        int lineCount = 0;

        try {
            br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null) {

                String[] lineValues = line.split(",");
                int length = lineValues.length;

                if (length != 7) {
                    throw new IOException("File has wrong format!");
                }

                //6th column is closing value
                values.add(Double.parseDouble(lineValues[5]));
                if (skipWeekends) {
                    DateTime newDate = DateTime.parse(lineValues[0] + " " + lineValues[1], dtf);
                    if (prevDate != null && Seconds.secondsBetween(prevDate, newDate).getSeconds() > period) {
                        weekendIndices.add(lineCount - 1);
                    }
                    prevDate = newDate;
                }
                lineCount++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        Object[] doubleVals = values.toArray();

        int finalLength = doubleVals.length - numberOfFeatures - 1;

        List<LabeledPoint> labeledPoints = new ArrayList<>();
        for (int i = 0; i < finalLength && labeledPoints.size() < numberOfPoints; i++) {
            //check if period contains weekend
            if (skipWeekends) {
                boolean skip = false;
                for (int j = i; j < i + numberOfFeatures - 1; j++) {
                    if (weekendIndices.contains(j)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }
            }

            double[] features = new double[numberOfFeatures];
            for (int j = 0; j < numberOfFeatures; j++) {
                features[j] = (double) doubleVals[i + j];
            }
            labeledPoints.add(new LabeledPoint((double) doubleVals[i + numberOfFeatures + 1], features));
        }

        System.out.println("Read " + labeledPoints.size() + " points.");

        return labeledPoints;
    }
    /*
    public static void write(String file, String separator, double[] firstCol, double[] secondCol) throws Exception {

        if (firstCol.length != secondCol.length) {
            throw new IllegalArgumentException("Lengths do not match!");
        }
        
        java.io.File courseCSV = new java.io.File(file);
        java.io.PrintWriter outfile = new java.io.PrintWriter(courseCSV);

        for (int i=0; i < firstCol.length ; i++) {
            outfile.println(String.valueOf(firstCol[i])+separator+String.valueOf(secondCol[i]));
        }

        outfile.close();
    }*/
}
