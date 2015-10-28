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
	
	// word translation
	TObjectIntHashMap<String>[] translation;
	public int enVocSize;
	
	public WordVector(Options options) throws IOException {
		this.options = options;
		langString = options.langString;
		langNum = langString.length;
		
		loadWordVector();
		loadTranslation();
	}
	
	public String constructFileName(int l) {
		return options.dataDir + "/universal_treebanks_v2.0/" + langString[l] + "-small5-embedding";
		//return options.dataDir + "/universal_treebanks_v2.0/" + langString[l] + "-en-embedding3";
	}
	
	public String constructTransFileName(int l) {
		return options.dataDir + "/universal_treebanks_v2.0/" + langString[l] + "-en.small5.pair";
		//return options.dataDir + "/universal_treebanks_v2.0/" + langString[l] + "-en.pair";
	}
	
	public void loadWordVector() throws IOException {
		System.out.print("load word vectors: " + constructFileName(0));
		wordVec = new double[langNum][][];
		dicts = new TObjectIntHashMap[langNum];
		
		for (int lang = 0; lang < langNum; ++lang) {
			System.out.print(" " + langString[lang] + " ");
			dicts[lang] = new TObjectIntHashMap<String>();
			
			//if (langString[lang].equals("ja"))
			//	continue;
			
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
	
	public void loadTranslation() throws IOException {
		System.out.print("load translation: " + constructTransFileName(0));
		
		translation = new TObjectIntHashMap[langNum];
		int enId = 0;
		for (int l = 0; l < langNum; ++l)
			if (langString[l].equals("en"))
				enId = l;
		translation[enId] = new TObjectIntHashMap<String>();
		
		for (int lang = 0; lang < langNum; ++lang) {
			if (langString[lang].equals("en"))
				continue;

			System.out.print(" " + langString[lang] + " ");
			translation[lang] = new TObjectIntHashMap<String>();
			BufferedReader br = new BufferedReader(new FileReader(constructTransFileName(lang)));
			String str = null;
			while ((str = br.readLine()) != null) {
				String[] data = str.split(" \\|\\|\\| ");
				String wen = data[1];
				String wde = data[0];
				//System.out.println(wen + " " + wde);
				int size = translation[enId].size();
				translation[enId].putIfAbsent(wen, size + 1);		// 1-based
				translation[lang].put(wde, translation[enId].get(wen));
			}
			br.close();
		}
		enVocSize = translation[enId].size();
		
		for (Object s : translation[enId].keys()) {
			Utils.Assert(translation[enId].get(s) <= enVocSize);
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
	
	public int getTranslationId(int lang, String w) {
		if (!translation[lang].containsKey(w))
			return -1;
		else
			return translation[lang].get(w) - 1;
	}
}