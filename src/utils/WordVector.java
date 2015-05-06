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
	public int size;
	
	public WordVector(Options options) throws IOException {
		this.options = options;
		langString = options.langString;
		langNum = langString.length;
		
		loadWordVector();
	}
	
	public String constructFileName(int l) {
		return options.dataDir + "/universal_treebanks_v2.0/" + langString[l] + "-multi-embedding";
	}
	
	public void loadWordVector() throws IOException {
		System.out.print("load word vectors: ");
		wordVec = new double[langNum][][];
		dicts = new TObjectIntHashMap[langNum];
		
		for (int lang = 0; lang < langNum; ++lang) {
			System.out.print(" " + langString[lang] + " ");
			dicts[lang] = new TObjectIntHashMap<String>();
			BufferedReader br = new BufferedReader(new FileReader(constructFileName(lang)));
			String[] data = br.readLine().split("\\s+");
			int wordNum = Integer.parseInt(data[0]);
			int dim = Integer.parseInt(data[1]);
			size = dim;
			
			wordVec[lang] = new double[wordNum][dim];
			for (int i = 0; i < wordNum; ++i) {
				data = br.readLine().split("\\s+");
				dicts[lang].put(data[0], i + 1);
				for (int l = 0; l < dim; ++l) {
					wordVec[lang][i][l] = Double.parseDouble(data[l + 1]);
				}
				Utils.normalize(wordVec[lang][i]);
			}
			br.close();
		}
		System.out.println("Done.");
	}
	
	public int getWordId(int lang, String w) {
		if (!dicts[lang].containsKey(w))
			return -1;
		else
			return dicts[lang].get(w) - 1;
	}
	
	public double[] getWordVec(int lang, int id) {
		return wordVec[lang][id];
	}
}
