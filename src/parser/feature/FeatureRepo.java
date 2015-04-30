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
	TypologicalInfo typo;
	int posNum;
	int labelNum;
	int d;
	int svoFeatureSize;
	int typoFeatureSize;
	
	FeatureVector[] dd;
	FeatureVector[] label;
	FeatureVector[] context;
	FeatureVector[] pos;
	FeatureVector[] svo;
	FeatureVector[] otherTypo;
	
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
			dd = new FeatureVector[2 * d];
			Utils.Assert(pn.node[2].featureSize == 1 + 2 * d);
			for (int i = 0; i < 2 * d; ++i) {
				dd[i] = ff.createDirDistFeatures(i, pn.node[2].featureSize, pn.node[2].featureBias);
			}
			
			if (options.learnLabel) {
				label = new FeatureVector[labelNum];
				Utils.Assert(pn.node[3].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					label[i] = ff.createLabelFeatures(i, pn.node[3].featureSize, pn.node[3].featureBias);
				}
			}
		}
		else if (options.tensorMode == TensorMode.Multiway) {
			pos = new FeatureVector[posNum];
			Utils.Assert(pn.node[0].featureSize == 1 + posNum);
			for (int p = 0; p < posNum; ++p) {
				pos[p] = ff.createPosFeatures(p, pn.node[0].featureSize, pn.node[0].featureBias);
				Utils.Assert(pn.node[0].featureSize == pn.node[1].featureSize);
			}
			
			Utils.Assert(pn.node[2].featureSize == 1 + posNum * 2);
			context = new FeatureVector[posNum * posNum];
			for (int pp = 0; pp < posNum; ++pp) {
				for (int np = 0; np < posNum; ++np) {
					context[pp * posNum + np]
							= ff.createContextPOSFeatures(pp, np, pn.node[2].featureSize, pn.node[2].featureBias);
					Utils.Assert(pn.node[2].featureSize == pn.node[3].featureSize);
				}
			}
			
			dd = new FeatureVector[2 * d];
			Utils.Assert(pn.node[4].featureSize == 1 + 2 * d);
			for (int i = 0; i < 2 * d; ++i) {
				dd[i] = ff.createDirDistFeatures(i, pn.node[4].featureSize, pn.node[4].featureBias);
			}
			
			if (options.learnLabel) {
				label = new FeatureVector[labelNum];
				Utils.Assert(pn.node[5].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					label[i] = ff.createLabelFeatures(i, pn.node[5].featureSize, pn.node[5].featureBias);
				}
			}
		}
		else if (options.tensorMode == TensorMode.Hierarchical) {
			ParameterNode delexical = pn;
			if (options.lexical) {
				delexical = pn.node[1];
			}
			
			context = new FeatureVector[posNum * posNum * typo.langNum];
			Utils.Assert(delexical.node[0].featureSize == 1 + posNum * typo.classNum * 2 + posNum * typo.familyNum * 2);
			for (int pp = 0; pp < posNum; ++pp) {
				for (int np = 0; np < posNum; ++np) {
					for (int l = 0; l < typo.langNum; ++l) {
						context[pp * posNum * typo.langNum + np * typo.langNum + l]
								= ff.createContextPOSFeatures(pp, np, l, delexical.node[0].featureSize, delexical.node[0].featureBias);
						Utils.Assert(delexical.node[0].featureSize == delexical.node[1].featureSize);
					}
				}
			}
			
			if (options.learnLabel) {
				ParameterNode arc = delexical.node[2];
				svo = new FeatureVector[2 * (2 * d) * typo.langNum];
				svoFeatureSize = arc.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						svo[i * typo.langNum + l] 
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_SBJ, i, l, arc.featureSize, arc.featureBias);
						svo[(2 * d) * typo.langNum + i * typo.langNum + l]
								= ff.createSVOFeatures(ff.POS_VERB, ff.POS_NOUN, ff.LABEL_DOBJ, i, l, arc.featureSize, arc.featureBias);
					}
				}

				label = new FeatureVector[labelNum];
				Utils.Assert(arc.node[0].featureSize == 1 + labelNum);
				for (int i = 0; i < labelNum; ++i) {
					label[i] = ff.createLabelFeatures(i, arc.node[0].featureSize,arc.node[0].featureBias);
				}
				
				ParameterNode t = arc.node[1];
				
				otherTypo = new FeatureVector[3 * (2 * d) * typo.langNum];
				typoFeatureSize = t.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						otherTypo[i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						otherTypo[(2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						otherTypo[2 * (2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_ADJ, i, l, t.featureSize, t.featureBias);
					}
				}
				
				pos = new FeatureVector[posNum];
				Utils.Assert(t.node[0].featureSize == 1 + posNum);
				for (int p = 0; p < posNum; ++p) {
					pos[p] = ff.createPosFeatures(p, t.node[0].featureSize, t.node[0].featureBias);
					Utils.Assert(t.node[0].featureSize == t.node[1].featureSize);
				}
				
				dd = new FeatureVector[2 * d * typo.langNum];
				Utils.Assert(t.node[2].featureSize == 1 + 2 * d * typo.classNum + 2 * d * typo.familyNum);
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						dd[i * typo.langNum + l] = ff.createDirDistTypoFeatures(i, l, t.node[2].featureSize, t.node[2].featureBias);
					}
				}
				
			}
			else {
				
				ParameterNode t = delexical.node[2];
				
				otherTypo = new FeatureVector[3 * (2 * d) * typo.langNum];
				typoFeatureSize = t.featureSize;
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						otherTypo[i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_ADP, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						otherTypo[(2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_NOUN, i, l, t.featureSize, t.featureBias);
						otherTypo[2 * (2 * d) * typo.langNum + i * typo.langNum + l] 
								= ff.createTypoFeatures(ff.POS_NOUN, ff.POS_ADJ, i, l, t.featureSize, t.featureBias);
					}
				}
				
				pos = new FeatureVector[posNum];
				Utils.Assert(t.node[0].featureSize == 1 + posNum);
				for (int p = 0; p < posNum; ++p) {
					pos[p] = ff.createPosFeatures(p, t.node[0].featureSize, t.node[0].featureBias);
					Utils.Assert(t.node[0].featureSize == t.node[1].featureSize);
				}
				
				dd = new FeatureVector[2 * d * typo.langNum];
				Utils.Assert(t.node[2].featureSize == 1 + 2 * d * typo.classNum + 2 * d * typo.familyNum);
				for (int i = 0; i < 2 * d; ++i) {
					for (int l = 0; l < typo.langNum; ++l) {
						dd[i * typo.langNum + l] = ff.createDirDistTypoFeatures(i, l, t.node[2].featureSize, t.node[2].featureBias);
					}
				}
				
			}
			
		}
		System.out.println("Done");
	}
	
	public FeatureVector getDDFv(int binDist) {
		return dd[binDist];
	}
	
	public FeatureVector getLabelFv(int l) {
		return label[l];
	}
	
	public FeatureVector getDDTypoFv(int binDist, int l) {
		return dd[binDist * typo.langNum + l];
	}
	
	public FeatureVector getContextFv(int pp, int np, int l) {
		return context[pp * posNum * typo.langNum + np * typo.langNum + l];
	}
	
	public FeatureVector getContextFv(int pp, int np) {
		return context[pp * posNum + np];
	}
	
	public FeatureVector getPOSFv(int p) {
		return pos[p];
	}
	
	public FeatureVector getSVOFv(int hp, int mp, int label, int binDist, int l) {
    	if (hp != ff.POS_VERB || (mp != ff.POS_NOUN && mp != ff.POS_PRON) 
    			|| (l != ff.LABEL_SBJ && l != ff.LABEL_SBJPASS && l != ff.LABEL_DOBJ && l != ff.LABEL_IOBJ))
    		return new FeatureVector(svoFeatureSize);
    	
    	if (l == ff.LABEL_SBJ || l == ff.LABEL_SBJPASS)
    		return svo[binDist * typo.langNum + l];
    	else
    		return svo[(2 * d) * typo.langNum + binDist * typo.langNum + l];
	}
	
	public FeatureVector getTypoFv(int hp, int mp, int binDist, int l) {
		if (hp == ff.POS_ADP && (mp == ff.POS_NOUN || mp == ff.POS_PRON)) {
			return otherTypo[binDist * typo.langNum + l];
		}
		else if (hp == ff.POS_NOUN && mp == ff.POS_NOUN) {
			return otherTypo[(2 * d) * typo.langNum + binDist * typo.langNum + l];
		}
		else if (hp == ff.POS_NOUN && mp == ff.POS_ADJ) {
			return otherTypo[2 * (2 * d) * typo.langNum + binDist * typo.langNum + l];
		}
		else {
			return new FeatureVector(typoFeatureSize);
		}
	}
}
