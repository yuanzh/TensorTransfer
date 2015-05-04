package utils;

import parser.Options;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.*;

public class WordVector {
	Options options;
	String[] langString;
	int langNum;
	double[][][] wordVec;	//[lang][wordid][dim]
	TObjectIntHashMap<String>[] dicts;
	
	public WordVector(Options options) throws IOException {
		this.options = options;
		langString = options.langString;
		langNum = langString.length;
		
		loadWordVector();
	}
	
	public void loadWordVector() throws IOException {
		wordVec = new double[langNum][][];
		dicts = new TObjectIntHashMap[langNum];
		
		for (int lang = 0; lang < langNum; ++lang) {
			
		}
	}
}
