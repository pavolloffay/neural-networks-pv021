/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.muni.neural.network.util;

import cz.muni.neural.network.Functions;
import cz.muni.neural.network.model.LabeledPoint;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 *
 * @author Václav
 */
public class CSVReader {
    
    public static List<LabeledPoint> read(String file, String separator, int numberOfPoints) throws IOException {
                  
	BufferedReader br = null;
	String line = "";
        List<LabeledPoint> labeledPoints = new ArrayList<>();
        
        int lineCount = 0;
        
	try {
            br = new BufferedReader(new FileReader(file));
            while ((line = br.readLine()) != null && lineCount < numberOfPoints) {

                String[] lineValues = line.split(separator);
                int length = lineValues.length;
                double[] doubleValues = new double[length];
                for (int i = 0; i < length; i++) {
                    doubleValues[i] = Double.parseDouble(lineValues[i]);
                }
                
                LabeledPoint labeledPoint = new LabeledPoint(doubleValues[doubleValues.length - 1], Arrays.copyOf(doubleValues, doubleValues.length-1));
                labeledPoints.add(labeledPoint);      
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
         
        System.out.println("Read " + labeledPoints.size() + " csv lines.");
        
        return labeledPoints;
    }
    
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
    }
}
