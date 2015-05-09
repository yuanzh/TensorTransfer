package utils;

import java.util.ArrayList;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;

public class Distribution {
	double[] targetBigram;
	double[] targetTrigram;
	double targetBigramSum;
	double targetTrigramSum;
	
	double[] sourceClassBigram;
	double[] sourceClassTrigram;
	double sourceClassBigramSum;
	double sourceClassTrigramSum;
	
	double[] sourceFamilyBigram;
	double[] sourceFamilyTrigram;
	double sourceFamilyBigramSum;
	double sourceFamilyTrigramSum;
	
	int langNum;
	int posNum;
	int targetLang;
	int langFamily;
	int langClass;
	boolean hasSameFamily;
	boolean hasSameClass;
	Options options;
	DependencyPipe pipe;
	TypologicalInfo typo;
	
	public Distribution(ArrayList<DependencyInstance> lstTrain, DependencyPipe pipe, Options options) {
		this.options = options;
		this.pipe = pipe;
		typo = pipe.typo;
		langNum = pipe.typo.langNum;
		posNum = pipe.poses.length;
		targetLang = options.targetLang;
		langFamily = typo.getFamily(targetLang);
		langClass = typo.getClass(targetLang);
		hasSameFamily = false;
		hasSameClass = true;
		
		targetBigram = new double[posNum * posNum];
		targetTrigram = new double[posNum * posNum * posNum];
		sourceClassBigram = new double[posNum * posNum];
		sourceClassTrigram = new double[posNum * posNum * posNum];
		sourceFamilyBigram = new double[posNum * posNum];
		sourceFamilyTrigram = new double[posNum * posNum * posNum];
		targetBigramSum = 0;
		targetTrigramSum = 0;
		sourceClassBigramSum = 0;
		sourceClassTrigramSum = 0;
		sourceFamilyBigramSum = 0;
		sourceFamilyTrigramSum = 0;
		
		for (int i = 0, L = lstTrain.size(); i < L; ++i) {
			addCount(lstTrain.get(i));
		}
		//normalize();
	}
	
	public void addCount(DependencyInstance inst) {
		addCount(inst, 1.0, false);
	}
	
	public void addCount(DependencyInstance inst, double scale, boolean addAsSource) {
		int n = inst.length;
		int lang = inst.lang;
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i == 0 ? pipe.ff.TOKEN_START : inst.postagids[i - 1];
			int np = i == n - 1 ? pipe.ff.TOKEN_END : inst.postagids[i + 1];
			
			if (!addAsSource && lang == targetLang) {
				targetBigram[p * posNum + pp] += scale;
				targetBigramSum += scale;
				targetTrigram[pp * posNum * posNum + p * posNum + np] += scale;
				targetTrigramSum += scale;
			}
			else {
				if (typo.getClass(lang) == langClass) {
					hasSameClass = true;
					sourceClassBigram[p * posNum + pp] += scale;
					sourceClassBigramSum += scale;
					sourceClassTrigram[pp * posNum * posNum + p * posNum + np] += scale;
					sourceClassTrigramSum += scale;
				}
				if (typo.getFamily(lang) == langFamily) {
					hasSameFamily = true;
					sourceFamilyBigram[p * posNum + pp] += scale;
					sourceFamilyBigramSum += scale;
					sourceFamilyTrigram[pp * posNum * posNum + p * posNum + np] += scale;
					sourceFamilyTrigramSum += scale;
				}
			}
		}
	}
	
	public void normalize() {
		Utils.normalize(targetBigram);
		Utils.normalize(targetTrigram);
		if (hasSameClass) {
			Utils.normalize(sourceClassBigram);
			Utils.normalize(sourceClassTrigram);
		}
		if (hasSameFamily) {
			Utils.normalize(sourceFamilyBigram);
			Utils.normalize(sourceFamilyTrigram);
		}
	}
	
	private double trans(double v) {
		return Math.log(v + 1e-4);
		//return v + 1e-4;
	}
	
	public double getScore(DependencyInstance inst) {
		Utils.Assert(inst.lang == targetLang);
		int n = inst.length;
		double score = 0.0;
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i == 0 ? pipe.ff.TOKEN_START : inst.postagids[i - 1];
			int np = i == n - 1 ? pipe.ff.TOKEN_END : inst.postagids[i + 1];
			
			int bigramCode = p * posNum + pp;
			int trigramCode = pp * posNum * posNum + p * posNum + np;
			
			//if (hasSameClass) {
				double pt = targetTrigram[trigramCode] / targetTrigramSum;
				double ps = sourceClassTrigramSum > 0.0 ? sourceClassTrigram[trigramCode] / sourceClassTrigramSum : 0.0;
				score += pt > ps ? trans(pt) - trans(ps) : 0.1 * (trans(pt) - trans(ps));
			//}
			
			//if (hasSameFamily) {
				pt = targetTrigram[trigramCode] / targetTrigramSum;
				ps = sourceFamilyTrigramSum > 0.0 ? sourceFamilyTrigram[trigramCode] / sourceFamilyTrigramSum : 0.0;
				score += pt > ps ? trans(pt) - trans(ps) : 0.1 * (trans(pt) - trans(ps));
			//}

			//pt = targetBigram[bigramCode] / targetBigramSum;
			//ps = sourceClassBigramSum > 0.0 ? sourceClassBigram[bigramCode] / sourceClassBigramSum : 0.0;
			//score += pt > ps ? 0.1 * (trans(pt) - trans(ps)) : 0.05 * (trans(pt) - trans(ps));

			//pt = targetBigram[bigramCode] / targetBigramSum;
			//ps = sourceFamilyBigramSum > 0.0 ? sourceFamilyBigram[bigramCode] / sourceFamilyBigramSum : 0.0;
			//score += pt > ps ? 0.1 * (trans(pt) - trans(ps)) : 0.05 * (trans(pt) - trans(ps));
		}
		return score;
	}
}
