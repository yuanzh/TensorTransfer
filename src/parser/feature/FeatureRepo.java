package parser.feature;

import parser.Options;
import parser.Options.TensorMode;
import parser.tensor.ParameterNode;
import utils.FeatureVector;
import utils.TypologicalInfo;
import utils.Utils;

public class FeatureRepo {
	// feature repository of some heavily reused feature vectors
	
	Options options;
	FeatureFactory ff;
	ParameterNode pn;
	transient TypologicalInfo typo;
	int posNum;
	int labelNum;
	int d;
	int svoFeatureSize;
	int typoFeatureSize;
	
	FeatureVector[] ddFv;
	FeatureVector[] labelFv;
	FeatureVector[] contextFv;
	FeatureVector[] posFv;
	FeatureVector[] svoFv;
	FeatureVector[] typoFv;
	
	FeatureVector emptyLabelFv;
	
	public FeatureRepo(Options options, FeatureFactory ff) {
		this.options = options;
		this.ff = ff;
		this.pn = ff.pn;
		typo = ff.typo;
		posNum = pn.posNum;
		labelNum = pn.labelNum;
		d = ParameterNode.d;
		
		buildRepository();
	}
	
	public void buildRepository() {
		System.out.print("Build tensor feature repository...");
		if (options.tensorMode == TensorMode.Threeway) {
			ddFv = new FeatureVector[2 * d];
			Utils.Assert(pn.node[2].featureSize == 1 + 2 * d);
			for (int i = 0; i < 2 * d; ++i) {
				ddFv[i] = ff.createDirDistFeatures(i, pn.node[2].featureSize, pn.node[2].featureBias);
			}
			
			if (options.learnLabel) {
				labelFv = new FeatureVector[labelNum];
				Utils.Assert(pn.node[3].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					labelFv[i] = ff.createLabelFeatures(i, pn.node[3].featureSize, pn.node[3].featureBias);
				}
				emptyLabelFv = new FeatureVector(pn.node[3].featureSize);
				emptyLabelFv.addEntry(0);
			}
		}
		else if (options.tensorMode == TensorMode.Multiway) {
			posFv = new FeatureVector[posNum];
			Utils.Assert(pn.node[0].featureSize == 1 + posNum);
			for (int p = 0; p < posNum; ++p) {
				posFv[p] = ff.createPosFeatures(p, pn.node[0].featureSize, pn.node[0].featureBias);
				Utils.Assert(pn.node[0].featureSize == pn.node[1].featureSize);
			}
			
			Utils.Assert(pn.node[2].featureSize == 1 + posNum * 2);
			contextFv = new FeatureVector[posNum * posNum];
			for (int pp = 0; pp < posNum; ++pp) {
				for (int np = 0; np < posNum; ++np) {
					contextFv[pp * posNum + np]
							= ff.createContextPOSFeatures(pp, np, pn.node[2].featureSize, pn.node[2].featureBias);
					Utils.Assert(pn.node[2].featureSize == pn.node[3].featureSize);
				}
			}
			
			ddFv = new FeatureVector[2 * d];
			Utils.Assert(pn.node[4].featureSize == 1 + 2 * d);
			for (int i = 0; i < 2 * d; ++i) {
				ddFv[i] = ff.createDirDistFeatures(i, pn.node[4].featureSize, pn.node[4].featureBias);
			}
			
			if (options.learnLabel) {
				labelFv = new FeatureVector[labelNum];
				Utils.Assert(pn.node[5].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					labelFv[i] = ff.createLabelFeatures(i, pn.node[5].featureSize, pn.node[5].featureBias);
				}
				emptyLabelFv = new FeatureVector(pn.node[5].featureSize);
				emptyLabelFv.addEntry(0);
			}
		}
		else if (options.tensorMode == TensorMode.Hierarchical) {
			ParameterNode delexical = pn;
			if (options.lexical) {
				delexical = pn.node[1];
			}
			
			contextFv = new FeatureVector[posNum * posNum * typo.langNum];
			//Utils.Assert(delexical.node[0].featureSize == 1 + posNum * typo.classNum * 2 + posNum * typo.familyNum * 2);
			for (int pp = 0; pp < posNum; ++pp) {
				for (int np = 0; np < posNum; ++np) {
					for (int l = 0; l < typo.langNum; ++l) {
						contextFv[pp * posNum * typo.langNum + np * typo.langNum + l]
								= ff.createContextPOSFeatures(pp, np, l, delexical.node[0].featureSize, delexical.node[0].featureBias);
						Utils.Assert(delexical.node[0].featureSize == delexical.node[1].featureSize);
					}
				}
			}
			
			if (options.learnLabel) {
				ParameterNode arc = delexical.node[2];
//				svo = new FeatureVector[2 * (2 * d) * typo.langNum];
//				svoFeatureSize = arc.featureSize;
//				for (int i = 0; i < 2 * d; ++i) {
//					for (int l = 0; l < typo.langNum; ++l) {
//						svo[i * typo.langNum + l] 
//								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_SBJ, i, l, arc.featureSize, arc.featureBias);
//						svo[(2 * d) * typo.langNum + i * typo.langNum + l]
//								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_DOBJ, i, l, arc.featureSize, arc.featureBias);
//					}
//				}
				svoFv = new FeatureVector[8 * (2 * d) * typo.langNum];
				svoFeatureSize = arc.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int lang = 0; lang < typo.langNum; ++lang) {
						// SV, NOUN, SBJ
						svoFv[i * typo.langNum + lang] 
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_SBJ, i, lang, arc.featureSize, arc.featureBias);
						// SV, NOUN, SBJPASS
						svoFv[(2 * d) * typo.langNum + i * typo.langNum + lang]
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_SBJPASS, i, lang, arc.featureSize, arc.featureBias);
						// SV, PRON, SBJ
						svoFv[2 * (2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_PRON, ff.LABEL_SBJ, i, lang, arc.featureSize, arc.featureBias);
						// SV, PRON, SBJPASS
						svoFv[3 * (2 * d) * typo.langNum + i * typo.langNum + lang]
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_PRON, ff.LABEL_SBJPASS, i, lang, arc.featureSize, arc.featureBias);
						// VO, NOUN, DOBJ
						svoFv[4 * (2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_DOBJ, i, lang, arc.featureSize, arc.featureBias);
						// VO, NOUN, IOBJ
						svoFv[5 * (2 * d) * typo.langNum + i * typo.langNum + lang]
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_IOBJ, i, lang, arc.featureSize, arc.featureBias);
						// VO, PRON, DOBJ
						svoFv[6 * (2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_PRON, ff.LABEL_DOBJ, i, lang, arc.featureSize, arc.featureBias);
						// VO, PRON, IOBJ
						svoFv[7 * (2 * d) * typo.langNum + i * typo.langNum + lang]
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_PRON, ff.LABEL_IOBJ, i, lang, arc.featureSize, arc.featureBias);
					}
				}
				

				labelFv = new FeatureVector[labelNum];
				Utils.Assert(arc.node[0].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					labelFv[i] = ff.createLabelFeatures(i, arc.node[0].featureSize,arc.node[0].featureBias);
				}
				emptyLabelFv = new FeatureVector(arc.node[0].featureSize);
				emptyLabelFv.addEntry(0);
				
				ParameterNode t = arc.node[1];
				
//				otherTypo = new FeatureVector[3 * (2 * d) * typo.langNum];
//				typoFeatureSize = t.featureSize;
//				for (int i = 0; i < 2 * d; ++i) {
//					for (int l = 0; l < typo.langNum; ++l) {
//						otherTypo[i * typo.langNum + l] 
//								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
//						otherTypo[(2 * d) * typo.langNum + i * typo.langNum + l] 
//								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
//						otherTypo[2 * (2 * d) * typo.langNum + i * typo.langNum + l] 
//								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_ADJ, i, l, t.featureSize, t.featureBias);
//					}
//				}
				
				typoFv = new FeatureVector[4 * (2 * d) * typo.langNum];
				typoFeatureSize = t.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int lang = 0; lang < typo.langNum; ++lang) {
						// Prep, ADP, NOUN
						typoFv[i * typo.langNum + lang] 
								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_NOUN, i, lang, t.featureSize, t.featureBias);
						// Prep, ADP, PRON
						typoFv[(2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_PRON, i, lang, t.featureSize, t.featureBias);
						// Gen
						typoFv[2 * (2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_NOUN, i, lang, t.featureSize, t.featureBias);
						// Adj
						typoFv[3 * (2 * d) * typo.langNum + i * typo.langNum + lang] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_ADJ, i, lang, t.featureSize, t.featureBias);
					}
				}

				posFv = new FeatureVector[posNum];
				Utils.Assert(t.node[0].featureSize == 1 + posNum);
				for (int p = 0; p < posNum; ++p) {
					posFv[p] = ff.createPosFeatures(p, t.node[0].featureSize, t.node[0].featureBias);
					Utils.Assert(t.node[0].featureSize == t.node[1].featureSize);
				}
				
				ddFv = new FeatureVector[2 * d * typo.langNum];
				//Utils.Assert(t.node[2].featureSize == 1 + 2 * d * typo.classNum + 2 * d * typo.familyNum);
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						ddFv[i * typo.langNum + l] = ff.createDirDistTypoFeatures(i, l, t.node[2].featureSize, t.node[2].featureBias);
					}
				}
				
			}
			else {
				
				ParameterNode t = delexical.node[2];
				
				typoFv = new FeatureVector[3 * (2 * d) * typo.langNum];
				typoFeatureSize = t.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						typoFv[i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						typoFv[(2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						typoFv[2 * (2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_ADJ, i, l, t.featureSize, t.featureBias);
					}
				}
				
				posFv = new FeatureVector[posNum];
				Utils.Assert(t.node[0].featureSize == 1 + posNum);
				for (int p = 0; p < posNum; ++p) {
					posFv[p] = ff.createPosFeatures(p, t.node[0].featureSize, t.node[0].featureBias);
					Utils.Assert(t.node[0].featureSize == t.node[1].featureSize);
				}
				
				ddFv = new FeatureVector[2 * d * typo.langNum];
				Utils.Assert(t.node[2].featureSize == 1 + 2 * d * typo.classNum + 2 * d * typo.familyNum);
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						ddFv[i * typo.langNum + l] = ff.createDirDistTypoFeatures(i, l, t.node[2].featureSize, t.node[2].featureBias);
					}
				}
				
			}
			
		}
		System.out.println("Done");
	}
	
	public FeatureVector getDDFv(int binDist) {
		return ddFv[binDist];
	}
	
	public FeatureVector getLabelFv(int label) {
		return label == -1 ? emptyLabelFv : labelFv[label];
	}
	
	public FeatureVector getDDTypoFv(int binDist, int lang) {
		return ddFv[binDist * typo.langNum + lang];
	}
	
	public FeatureVector getContextFv(int pp, int np, int lang) {
		return contextFv[pp * posNum * typo.langNum + np * typo.langNum + lang];
	}
	
	public FeatureVector getContextFv(int pp, int np) {
		return contextFv[pp * posNum + np];
	}
	
	public FeatureVector getPOSFv(int p) {
		return posFv[p];
	}
	
	public FeatureVector getSVOFv(int hp, int mp, int label, int binDist, int lang) {
    	if (hp != ff.POS_VERB || (mp != ff.POS_NOUN && mp != ff.POS_PRON) 
    			|| (label != ff.LABEL_SBJ && label != ff.LABEL_SBJPASS && label != ff.LABEL_DOBJ && label != ff.LABEL_IOBJ)) {
    		//System.out.println("lalala");
    		return new FeatureVector(svoFeatureSize);
    	}
    	
		//System.out.println("yayaya");
    	//if (l == ff.LABEL_SBJ || l == ff.LABEL_SBJPASS)
    	//	return svo[binDist * typo.langNum + l];
    	//else
    	//	return svo[(2 * d) * typo.langNum + binDist * typo.langNum + l];
    	
		// SV, NOUN, SBJ
		if (mp == ff.POS_NOUN && label == ff.LABEL_SBJ)
			return svoFv[binDist * typo.langNum + lang]; 
		// SV, NOUN, SBJPASS
		else if (mp == ff.POS_NOUN && label == ff.LABEL_SBJPASS)
			return svoFv[(2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// SV, PRON, SBJ
		else if (mp == ff.POS_PRON && label == ff.LABEL_SBJ)
			return svoFv[2 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// SV, PRON, SBJPASS
		else if (mp == ff.POS_PRON && label == ff.LABEL_SBJPASS)
			return svoFv[3 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// VO, NOUN, DOBJ
		else if (mp == ff.POS_NOUN && label == ff.LABEL_DOBJ)
			return svoFv[4 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// VO, NOUN, IOBJ
		else if (mp == ff.POS_NOUN && label == ff.LABEL_IOBJ)
			return svoFv[5 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// VO, PRON, DOBJ
		else if (mp == ff.POS_PRON && label == ff.LABEL_DOBJ)
			return svoFv[6 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		// VO, PRON, IOBJ
		else if (mp == ff.POS_PRON && label == ff.LABEL_IOBJ)
			return svoFv[7 * (2 * d) * typo.langNum + binDist * typo.langNum + lang]; 
		else {
			Utils.ThrowException("bad getSOV");
			return null;
		}
	}
	
	public FeatureVector getTypoFv(int hp, int mp, int binDist, int lang) {
//		if (hp == ff.POS_ADP && (mp == ff.POS_NOUN || mp == ff.POS_PRON)) {
//			return otherTypo[binDist * typo.langNum + l];
//		}
//		else if (hp == ff.POS_NOUN && mp == ff.POS_NOUN) {
//			return otherTypo[(2 * d) * typo.langNum + binDist * typo.langNum + l];
//		}
//		else if (hp == ff.POS_NOUN && mp == ff.POS_ADJ) {
//			return otherTypo[2 * (2 * d) * typo.langNum + binDist * typo.langNum + l];
//		}
//		else {
//			return new FeatureVector(typoFeatureSize);
//		}
		
		if (hp == ff.POS_ADP && mp == ff.POS_NOUN) {
			return typoFv[binDist * typo.langNum + lang];
		}
		else if (hp == ff.POS_ADP && mp == ff.POS_PRON) {
			return typoFv[(2 * d) * typo.langNum + binDist * typo.langNum + lang];
		}
		else if (hp == ff.POS_NOUN && mp == ff.POS_NOUN) {
			return typoFv[2 * (2 * d) * typo.langNum + binDist * typo.langNum + lang];
		}
		else if (hp == ff.POS_NOUN && mp == ff.POS_ADJ) {
			return typoFv[3 * (2 * d) * typo.langNum + binDist * typo.langNum + lang];
		}
		else {
			return new FeatureVector(typoFeatureSize);
		}
	}
}
