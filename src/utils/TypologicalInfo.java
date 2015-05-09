package utils;

import parser.Options;
import java.io.*;

public class TypologicalInfo {

	public enum TypoFeatureType {
		SV,
		VO,
		Prep,
		Gen,
		Adj,
		Count,
	}

	Options options;
	
	public int langNum;
	public int featureNum;
	public int classNum;
	public int familyNum;
	public int bit;
	
	public int[] lang2Class;
	public int[] lang2Family;
	public int[][] lang2Feature;
	public int[] numberOfValues;
	
	public double[][] typoVec;
	
	public TypologicalInfo(Options options) throws IOException {
		this.options = options;
		loadData(options.typoFile);
	}
	
	public void loadData(String file) throws IOException {
		System.out.println("load typo information from " + file);
		BufferedReader br = new BufferedReader(new FileReader(file));
		String[] data = br.readLine().split("\\s+");
		langNum = Integer.parseInt(data[0]);
		featureNum = Integer.parseInt(data[1]);
		classNum = Integer.parseInt(data[2]);
		familyNum = Integer.parseInt(data[3]);
		
		Utils.Assert(featureNum == TypoFeatureType.Count.ordinal());
		
		data = br.readLine().split("\\s+");
		numberOfValues = new int[featureNum];
		Utils.Assert(data.length == featureNum);
		for (int i = 0; i < featureNum; ++i)
			numberOfValues[i] = Integer.parseInt(data[i]);
		
		//bit = Utils.log2(familyNum + classNum + 1);
		bit = Utils.log2(langNum + classNum + 1);
		
		Utils.Assert(langNum == options.langString.length);
		lang2Class = new int[langNum];
		lang2Family = new int[langNum];
		lang2Feature = new int[langNum][featureNum];
		typoVec = new double[langNum][langNum];
		
		for (int i = 0; i < langNum; ++i) {
			data = br.readLine().split("\\s+");
			Utils.Assert(options.langString[i].equals(data[0]) && data.length == featureNum + 2 + 1);
			for (int j = 0; j < featureNum; ++j) {
				lang2Feature[i][j] = Integer.parseInt(data[j + 1]);
			}
			lang2Class[i] = Integer.parseInt(data[1 + featureNum]);
			lang2Family[i] = Integer.parseInt(data[1 + featureNum + 1]);
		}
		
		for (int i = 0; i < langNum; ++i) {
			data = br.readLine().trim().split("\\s+");
			Utils.Assert(data.length == langNum);
			for (int j = 0; j < langNum; ++j)
				typoVec[i][j] = Double.parseDouble(data[j]);
			//System.out.println(options.langString[i] + " " + typoVec[i][0]);
			Utils.normalize(typoVec[i]);
		}
		br.close();
	}
	
	public int[] getFeature(int l) {
		return lang2Feature[l];
	}
	
	public int getClass(int l) {
		return lang2Class[l];
	}
	
	public int getFamily(int l) {
		return lang2Family[l];
	}
	
	public int getNumberOfValues(TypoFeatureType type) {
		return numberOfValues[type.ordinal()];
	}
	
	public double[] getTypoVec(int l) {
		return typoVec[l];
	}
}
