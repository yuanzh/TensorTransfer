package parser.feature;

import static parser.feature.FeatureTemplate.Arc.numArcFeatBits;

import java.io.Serializable;

import gnu.trove.set.hash.TLongHashSet;
import parser.DependencyInstance;
import parser.Options;
import parser.Options.TensorMode;
import parser.feature.FeatureTemplate.Arc;
import parser.tensor.ParameterNode;
import utils.Alphabet;
import utils.FeatureVector;
import utils.TypologicalInfo;
import utils.Utils;
import utils.TypologicalInfo.TypoFeatureType;

public class FeatureFactory implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	Options options;
	
	public int TOKEN_START = 1;
	public int TOKEN_END = 2;
	public int TOKEN_MID = 3;
	public int POS_NOUN = 4;
	public int POS_PRON = 5;
	public int POS_ADJ = 6;
	public int POS_VERB = 7;
	public int POS_ADP = 8;
	public int LABEL_SBJ = 2;
	public int LABEL_DOBJ = 3;
	public int LABEL_IOBJ = 4;
	public int LABEL_SBJPASS = 5;
	
	public int tagNumBits, depNumBits, flagBits;

	public int posNum, labelNum;
	public TypologicalInfo typo;
	public FeatureRepo fr;
    public ParameterNode pn;

    public Alphabet arcAlphabet;
    
	public int numArcFeats;	// number of arc structure features

	public FeatureFactory(Options options) {
		this.options = options;
		arcAlphabet = new Alphabet();
		numArcFeats = 0;
	}

	public void closeAlphabets()
	{
		arcAlphabet.stopGrowth();
	}

	public void initFeatureAlphabets(DependencyInstance inst) {
		// add CRF features
		initCRFFeatureVector(inst);
		
		// tensor features
		if (options.tensorMode == TensorMode.Threeway)
			initThreewayFeatureAlphabets(inst);
		else if (options.tensorMode == TensorMode.Multiway)
			initMultiwayFeatureAlphabets(inst);
		else if (options.tensorMode == TensorMode.Hierarchical)
			initHierarchicalFeatureAlphabets(inst);
		else
			Utils.ThrowException("not supported structure");
	}
	
	public void initCRFFeatureVector(DependencyInstance inst) {
		int n = inst.length;
		for (int i = 1; i < n; ++i) {
			createArcFeatures(inst, inst.heads[i], i);
			createArcLabelFeatures(inst, inst.heads[i], i, inst.deplbids[i]);
		}
	}
	
	public void initThreewayFeatureAlphabets(DependencyInstance inst) {
		// head and modifier
		int n = inst.length;
		
		for (int i = 1; i < n; ++i) {
			int h = inst.heads[i];
			FeatureVector fv = createThreewayPosFeatures(inst, h, pn.node[0].featureSize, pn.node[0].featureBias);
			pn.node[0].setActiveFeature(fv);
			
			fv = createThreewayPosFeatures(inst, i, pn.node[0].featureSize, pn.node[0].featureBias);
			pn.node[1].setActiveFeature(fv);	// root cannot be the modifier

			fv = fr.getDDFv(getBinnedDistance(h - i));
			pn.node[2].setActiveFeature(fv);

			if (options.learnLabel) {
				fv = fr.getLabelFv(inst.deplbids[i]);
				pn.node[3].setActiveFeature(fv);
			}
		}
	}
	
	public void initMultiwayFeatureAlphabets(DependencyInstance inst) {
		// head and modifier
		int n = inst.length;
		
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i > 0 ? inst.postagids[i - 1] : TOKEN_START;
			int np = i < n - 1 ? inst.postagids[i + 1] : TOKEN_END;

			FeatureVector fv = fr.getContextFv(pp, np);
			pn.node[2].setActiveFeature(fv);
			
			if (i > 0) {
				int h = inst.heads[i];
				int hp = inst.postagids[h];
				int binDist = getBinnedDistance(h - i);
				
				fv = fr.getPOSFv(hp);
				pn.node[0].setActiveFeature(fv);
				
				fv = fr.getPOSFv(p);
				pn.node[1].setActiveFeature(fv);
				
				fv = fr.getContextFv(pp, np);
				pn.node[3].setActiveFeature(fv);
				
				fv = fr.getDDFv(binDist);
				pn.node[4].setActiveFeature(fv);

				if (options.learnLabel) {
					fv = fr.getLabelFv(inst.deplbids[i]);
					pn.node[5].setActiveFeature(fv);
				}
			}

		}
	}
	
	public void initHierarchicalFeatureAlphabets(DependencyInstance inst) {
		int n = inst.length;
		ParameterNode delexical = pn;
		if (options.lexical) {
			// TODO: set lexical features
			delexical = pn.node[1];
		}
		
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i > 0 ? inst.postagids[i - 1] : TOKEN_START;
			int np = i < n - 1 ? inst.postagids[i + 1] : TOKEN_END;
			
			FeatureVector fv = fr.getContextFv(pp, np, inst.lang);
			delexical.node[0].setActiveFeature(fv);
			if (i > 0) {
				int h = inst.heads[i];
				int hp = inst.postagids[h];
				int binDist = getBinnedDistance(h - i);
				
				delexical.node[1].setActiveFeature(fv);
				if (options.learnLabel) {
					int label = inst.deplbids[i];
					
					ParameterNode arc = delexical.node[2];
					fv = fr.getSVOFv(hp, p, label, binDist, inst.lang);
					arc.setActiveFeature(fv);
					
					fv = fr.getLabelFv(label);
					arc.node[0].setActiveFeature(fv);
					
					ParameterNode typo = arc.node[1];
					fv = fr.getTypoFv(hp, p, binDist, inst.lang);
					typo.setActiveFeature(fv);
					
					fv = fr.getPOSFv(hp);
					typo.node[0].setActiveFeature(fv);
					fv = fr.getPOSFv(p);
					typo.node[1].setActiveFeature(fv);
					fv = fr.getDDTypoFv(binDist, inst.lang);
					typo.node[2].setActiveFeature(fv);
				}
				else {
					ParameterNode typo = delexical.node[2];
					fv = fr.getTypoFv(hp, p, binDist, inst.lang);
					typo.setActiveFeature(fv);
					
					fv = fr.getPOSFv(hp);
					typo.node[0].setActiveFeature(fv);
					fv = fr.getPOSFv(p);
					typo.node[1].setActiveFeature(fv);
					fv = fr.getDDTypoFv(binDist, inst.lang);
					typo.node[2].setActiveFeature(fv);
				}
			}
			
		}
	}
	
	/***
	 * tensor features
	 */

    public FeatureVector createThreewayPosFeatures(DependencyInstance inst, int i, int dim, int[] bias) 
    {
    	int[] pos = inst.postagids;
        
        FeatureVector fv = new FeatureVector(dim);
    	
        int p0 = pos[i];
    	int pLeft = i > 0 ? pos[i-1] : TOKEN_START;
    	int pRight = i < pos.length-1 ? pos[i+1] : TOKEN_END;
    	
    	Utils.Assert(p0 < posNum);

    	int code = 0;
        
    	// bias
    	code = 0;
    	fv.addEntry(code);

    	// p0, p1, p-1
    	code = bias[0] + p0;
    	fv.addEntry(code);
    	code = bias[1] + pLeft;
    	fv.addEntry(code);
    	code = bias[2] + pRight;
    	fv.addEntry(code);
    	
    	// (p-1,p0), (p0,p1)
    	code = bias[3] + getPPCode(pLeft, p0);
    	fv.addEntry(code);
    	code = bias[4] + getPPCode(p0, pRight);
    	fv.addEntry(code);
    	
    	return fv;
    }
    
    public FeatureVector createDirDistFeatures(int binDist, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	
    	int code = 0;
    	fv.addEntry(code);
    	fv.addEntry(bias[0] + binDist);
    	
    	return fv;
    }
    
    public FeatureVector createDirDistTypoFeatures(int binDist, int lang, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	
    	int code = 0;
    	int d = ParameterNode.d;
    	fv.addEntry(code);

    	int c = typo.getClass(lang);
    	code = c * 2 * d + binDist;
    	fv.addEntry(bias[0] + code);
    	
    	int f = typo.getFamily(lang);
    	code = f * 2 * d + binDist;
    	fv.addEntry(bias[1] + code);
    	
    	return fv;
    }
    
    public FeatureVector createLabelFeatures(int label, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	
    	int code = 0;
    	fv.addEntry(code);
    	fv.addEntry(bias[0] + label);
    	
    	return fv;
    }
    
    public FeatureVector createPosFeatures(int p, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	
    	int code = 0;
    	fv.addEntry(code);
    	fv.addEntry(bias[0] + p);
    	
    	return fv;
    }
    
    public FeatureVector createContextPOSFeatures(int pp, int np, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	int code = 0;
    	fv.addEntry(code);
    	
    	code = pp;
    	fv.addEntry(bias[0] + code);
    	code = np;
    	fv.addEntry(bias[1] + code);
    	
    	return fv;
    }
    
    public FeatureVector createContextPOSFeatures(int pp, int np, int lang, int dim, int[] bias) {
    	FeatureVector fv = new FeatureVector(dim);
    	int code = 0;
    	fv.addEntry(code);
    	
    	int c = typo.getClass(lang);
    	code = c * posNum + pp;
    	fv.addEntry(bias[0] + code);
    	code = c * posNum + np;
    	fv.addEntry(bias[1] + code);
    	
    	int f = typo.getFamily(lang);
    	code = f * posNum + pp;
    	fv.addEntry(bias[2] + code);
    	code = f * posNum + np;
    	fv.addEntry(bias[3] + code);
    	
    	return fv;
    }
    
    public FeatureVector createSVOFeatures(int hp, int mp, int l, int binDist, int lang, int dim, int[] bias) {
    	// no bias feature
    	
    	FeatureVector fv = new FeatureVector(dim);
    	if (hp != POS_VERB || (mp != POS_NOUN && mp != POS_PRON) 
    			|| (l != LABEL_SBJ && l != LABEL_SBJPASS && l != LABEL_DOBJ && l != LABEL_IOBJ))
    		Utils.ThrowException("should not go here");
    	
    	int code = 0;
    	int d = ParameterNode.d;
    	if (l == LABEL_SBJ || l == LABEL_SBJPASS) {
    		int v = typo.getFeature(lang)[TypoFeatureType.SV.ordinal()];
    		int offset = 2 + 2 * d;
    		
    		boolean leftArc = binDist < d ? true : false;
    		code = binDist < d ? 0 : 1;		// 0: left
    		fv.addEntry(v * offset + code);
    		//code = 2 + binDist;
    		//fv.addEntry(v * offset + code);
    		
    		if (v == 0) {
    			// SV
    			code = leftArc ? 0 : 1;	// 1: violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    		else if (v == 1) {
    			// VS
    			code = leftArc ? 1 : 0;	// 1: violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    		else {
    			code = 0;		// not violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    	}
    	else {
    		int v = typo.getFeature(lang)[TypoFeatureType.VO.ordinal()];
    		int offset = 2 + 2 * d;

    		boolean leftArc = binDist < d ? true : false;
    		code = binDist < d ? 0 : 1;		// 0: left
    		fv.addEntry(bias[1] + v * offset + code);
    		//code = 2 + binDist;
    		//fv.addEntry(bias[1] + v * offset + code);
    		
    		if (v == 0) {
    			// VO
    			code = leftArc ? 1 : 0;	// 1: violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    		else if (v == 1) {
    			// OV
    			code = leftArc ? 0 : 1;	// 1: violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    		else {
    			code = 0;		// not violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    	}
    	
    	return fv;
    }
    
    public FeatureVector createTypoFeatures(int hp, int mp, int binDist, int lang, int dim, int[] bias) {
    	// no bias featuer
    	
    	FeatureVector fv = new FeatureVector(dim);
    	
    	int code = 0;
    	int d = ParameterNode.d;
    	
    	if (hp == POS_ADP && (mp == POS_NOUN || mp == POS_PRON)) {
    		int v = typo.getFeature(lang)[TypoFeatureType.Prep.ordinal()];
    		int offset = 2 + 2 * d;
    		
    		boolean leftArc = binDist < d ? true : false;
    		code = binDist < d ? 0 : 1;		// 0: left
    		fv.addEntry(v * offset + code);
    		//code = 2 + binDist;
    		//fv.addEntry(v * offset + code);
    		
    		if (v == 0) {
    			// Prep
    			code = leftArc ? 1 : 0;	// 1: violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    		else if (v == 1) {
    			// PostP
    			code = leftArc ? 0 : 1;	// 1: violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    		else {
    			code = 0;		// not violate
    			fv.addEntry(bias[0] + code);
    			//code = 2 + binDist;
    			//fv.addEntry(bias[0] + code);
    		}
    	}
    	else if (hp == POS_NOUN && mp == POS_NOUN) {
    		int v = typo.getFeature(lang)[TypoFeatureType.Gen.ordinal()];
    		int offset = 2 + 2 * d;
    		
    		boolean leftArc = binDist < d ? true : false;
    		code = binDist < d ? 0 : 1;		// 0: left
    		fv.addEntry(bias[1] + v * offset + code);
    		//code = 2 + binDist;
    		//fv.addEntry(bias[1] + v * offset + code);
    		
    		if (v == 0) {
    			// Gen-Noun
    			code = leftArc ? 0 : 1;	// 1: violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    		else if (v == 1) {
    			// Noun-Gen
    			code = leftArc ? 1 : 0;	// 1: violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    		else {
    			code = 0;		// not violate
    			fv.addEntry(bias[2] + code);
    			//code = 2 + binDist;
    			//fv.addEntry(bias[2] + code);
    		}
    	}
    	else if (hp == POS_NOUN && mp == POS_ADJ) {
    		int v = typo.getFeature(lang)[TypoFeatureType.Adj.ordinal()];
    		int offset = 2 + 2 * d;
    		
    		boolean leftArc = binDist < d ? true : false;
    		code = binDist < d ? 0 : 1;		// 0: left
    		fv.addEntry(bias[3] + v * offset + code);
    		//code = 2 + binDist;
    		//fv.addEntry(bias[3] + v * offset + code);
    		
    		if (v == 0) {
    			// Adj-Noun
    			code = leftArc ? 0 : 1;	// 1: violate
    			fv.addEntry(bias[4] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[4] + code);
    		}
    		else if (v == 1) {
    			// Noun-Adj
    			code = leftArc ? 1 : 0;	// 1: violate
    			fv.addEntry(bias[4] + code);
    			//code = 2 + code * 2 * d + binDist;
    			//fv.addEntry(bias[4] + code);
    		}
    		else {
    			code = 0;		// not violate
    			fv.addEntry(bias[4] + code);
    			//code = 2 + binDist;
    			//fv.addEntry(bias[4] + code);
    		}
    	}
    	
    	return fv;
    }
    
    /***
     * traditional features
     */
    
    public FeatureVector createArcFeatures(DependencyInstance inst, int h, int m) 
    {
    	FeatureVector fv = new FeatureVector(numArcFeats);
    	
    	//addDelexicalFeatures(inst, h, m, 0, fv);
    	addDelexicalCFFeatures(inst, h, m, 0, fv);
    	addBareFeatures(inst, h, m, 0, fv);
    	addSelectiveFeatures(inst, h, m, 0, fv);
    	
    	return fv;
    }
    
    public FeatureVector createArcLabelFeatures(DependencyInstance inst, int h, int m, int label) 
    {
    	FeatureVector fv = new FeatureVector(numArcFeats);
    	if (!options.learnLabel) 
    		return fv;
    	
    	//addDelexicalFeatures(inst, h, m, label + 1, fv);
    	addDelexicalCFFeatures(inst, h, m, label + 1, fv);
    	addBareFeatures(inst, h, m, label + 1, fv);
    	addSelectiveFeatures(inst, h, m, label + 1, fv);
    	
    	return fv;
    }
    
    public void addDelexicalFeatures(DependencyInstance inst, int h, int m, int label, FeatureVector fv) {
	    long code = 0;
		int[] pos = inst.postagids;
		
		int tid = label << 4;
		int attDist = getBinnedDistance(h - m) + 1;
		int n = inst.length;
		
		int HP = pos[h] + 1;
		int MP = pos[m] + 1;
		int HPp = (h > 0 ? pos[h - 1] : TOKEN_START) + 1;
		int HPn = (h < n - 1 ? pos[h + 1] : TOKEN_END) + 1;
		int MPp = (m > 0 ? pos[m - 1] : TOKEN_START) + 1;
		int MPn = (m < n - 1 ? pos[m + 1] : TOKEN_END) + 1;
		
    	code = createArcCodeP(Arc.ATTDIST, 0) | tid;
    	addArcFeature(code | attDist, fv);
    	
    	code = createArcCodeP(Arc.HP, HP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodeP(Arc.MP, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePP(Arc.HP_MP, HP, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPP(Arc.HPp_HP_MP, HPp, HP, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPP(Arc.HP_HPn_MP, HP, HPn, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPP(Arc.HP_MPp_MP, HP, MPp, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPP(Arc.HP_MP_MPn, HP, MP, MPn) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPPP(Arc.HPp_HP_MP_MPn, HPp, HP, MP, MPn) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPPP(Arc.HP_HPn_MP_MPn, HP, HPn, MP, MPn) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPPP(Arc.HP_HPn_MPp_MP, HP, HPn, MPp, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	code = createArcCodePPPP(Arc.HPp_HP_MPp_MP, HPp, HP, MPp, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | attDist, fv);

    	int large = Math.max(h, m);
    	int small = Math.min(h, m);
    	
    	for (int i = small + 1; i < large; ++i) {
    		int BP = pos[i];
        	code = createArcCodePPP(Arc.HP_BP_MP, HP, BP, MP) | tid;
        	addArcFeature(code, fv);
    	}

    }
    
    public void addDelexicalCFFeatures(DependencyInstance inst, int h, int m, int label, FeatureVector fv) {
	    long code = 0;
		int[] pos = inst.postagids;
		
		int tid = label << 4;
		int attDist = getBinnedDistance(h - m) + 1;
		int c = (typo.getClass(inst.lang) + 1) << flagBits;
		int f = (typo.getFamily(inst.lang) + typo.classNum + 1) << flagBits;
		int n = inst.length;
		
		int HP = pos[h] + 1;
		int MP = pos[m] + 1;
		int HPp = (h > 0 ? pos[h - 1] : TOKEN_START) + 1;
		int HPn = (h < n - 1 ? pos[h + 1] : TOKEN_END) + 1;
		int MPp = (m > 0 ? pos[m - 1] : TOKEN_START) + 1;
		int MPn = (m < n - 1 ? pos[m + 1] : TOKEN_END) + 1;
		
    	code = (createArcCodeP(Arc.ATTDIST, 0) << tagNumBits) | tid;
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);
    	
    	code = (createArcCodeP(Arc.HP, HP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodeP(Arc.MP, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePP(Arc.HP_MP, HP, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPP(Arc.HPp_HP_MP, HPp, HP, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPP(Arc.HP_HPn_MP, HP, HPn, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPP(Arc.HP_MPp_MP, HP, MPp, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPP(Arc.HP_MP_MPn, HP, MP, MPn) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPPP(Arc.HPp_HP_MP_MPn, HPp, HP, MP, MPn) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPPP(Arc.HP_HPn_MP_MPn, HP, HPn, MP, MPn) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPPP(Arc.HP_HPn_MPp_MP, HP, HPn, MPp, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	code = (createArcCodePPPP(Arc.HPp_HP_MPp_MP, HPp, HP, MPp, MP) << tagNumBits) | tid;
    	addArcFeature(code | c, fv);
    	addArcFeature(code | f, fv);
    	addArcFeature(code | c | attDist, fv);
    	addArcFeature(code | f | attDist, fv);

    	int large = Math.max(h, m);
    	int small = Math.min(h, m);
    	
    	for (int i = small + 1; i < large; ++i) {
    		int BP = pos[i];
        	code = (createArcCodePPP(Arc.HP_BP_MP, HP, BP, MP) << tagNumBits) | tid;
        	addArcFeature(code | c, fv);
        	addArcFeature(code | f, fv);
    	}
    }
    
    public void addBareFeatures (DependencyInstance inst, int h, int m, int label, FeatureVector fv) {
	    long code = 0;
		int[] pos = inst.postagids;
		
		int tid = label << 4;
		int dist = getBinnedDistance(Math.abs(h - m)) + 1;
		
		int HP = pos[h] + 1;
		int MP = pos[m] + 1;

    	code = createArcCodeP(Arc.DIST, 0) | tid;
    	addArcFeature(code | dist, fv);
    	
    	code = createArcCodeP(Arc.B_HP, HP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | dist, fv);

    	code = createArcCodeP(Arc.B_MP, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | dist, fv);

    	code = createArcCodePP(Arc.B_HP_MP, HP, MP) | tid;
    	addArcFeature(code, fv);
    	addArcFeature(code | dist, fv);
    }
    
    public void addSelectiveFeatures (DependencyInstance inst, int h, int m, int label, FeatureVector fv) {
	    long code = 0;
		int[] pos = inst.postagids;
		int[] feature = typo.getFeature(inst.lang);
		
		int tid = label << 4;
		int dir = h < m ? 1 : 2;
		
		int HP = pos[h];
		int MP = pos[m];

	    if (HP == POS_VERB && MP == POS_NOUN && 
	    	(label == LABEL_SBJ || label == LABEL_SBJPASS)) {
		    code = createArcCodeP(Arc.SV_NOUN, feature[TypoFeatureType.SV.ordinal()] + 1) | tid;
	    	addArcFeature(code | dir, fv);
	    }
    	
	    if (HP == POS_VERB && MP == POS_PRON && 
		    	(label == LABEL_SBJ || label == LABEL_SBJPASS)) {
	    	code = createArcCodeP(Arc.SV_PRON, feature[TypoFeatureType.SV.ordinal()] + 1) | tid;
	    	addArcFeature(code | dir, fv);
	    }
	    	
	    if (HP == POS_VERB && MP == POS_NOUN && 
		    	(label == LABEL_DOBJ || label == LABEL_IOBJ)) {
	    	code = createArcCodeP(Arc.VO_NOUN, feature[TypoFeatureType.VO.ordinal()]) | tid;
	    	addArcFeature(code | dir, fv);
	    }
	    	
	    if (HP == POS_VERB && MP == POS_PRON && 
		    	(label == LABEL_DOBJ || label == LABEL_IOBJ)) {
	    	code = createArcCodeP(Arc.VO_PRON, feature[TypoFeatureType.VO.ordinal()]) | tid;
	    	addArcFeature(code | dir, fv);
	    }
	    	
	    if (HP == POS_ADP && MP == POS_NOUN) {
			    code = createArcCodeP(Arc.ADP_NOUN, feature[TypoFeatureType.Prep.ordinal()]) | tid;
		    	addArcFeature(code | dir, fv);
		    }
	    	
	    if (HP == POS_ADP && MP == POS_PRON) {
			    code = createArcCodeP(Arc.ADP_PRON, feature[TypoFeatureType.Prep.ordinal()]) | tid;
		    	addArcFeature(code | dir, fv);
		    }
	    	
	    if (HP == POS_NOUN && MP == POS_NOUN) {
			    code = createArcCodeP(Arc.GEN, feature[TypoFeatureType.Gen.ordinal()]) | tid;
		    	addArcFeature(code | dir, fv);
		    }
	    	
	    if (HP == POS_NOUN && MP == POS_ADJ) {
			    code = createArcCodeP(Arc.ADJ, feature[TypoFeatureType.Adj.ordinal()]) | tid;
		    	addArcFeature(code | dir, fv);
		    }
    }
    
    /***
     * generate code
     */
    
    public final int getBinnedDistance(int x) {
    	// x = h-c, 0-6: left, 7-13:right
    	
    	int flag = 0;
    	int add = 0;
    	if (x < 0) {
    		x = -x;
    		//flag = 8;
    		add = 7;
    	}
    	if (x > 10)          // x > 10
    		flag |= 0x7;
    	else if (x > 5)		 // x = 6 .. 10
    		flag |= 0x6;
    	else
    		flag |= x;   	 // x = 1 .. 5
    	return flag+add-1;	// zero based
    }
    
    public int getPPCode(int p0, int p1) {
    	return p0 * posNum + p1;
    }

	public long createArcCodeP(FeatureTemplate.Arc temp, long x)
	{
		return ((x << numArcFeatBits) | temp.ordinal()) << flagBits;
	}
	
	public long createArcCodePP(FeatureTemplate.Arc temp, long x, long y)
	{
		return ((((x << tagNumBits) | y) << numArcFeatBits) | temp.ordinal()) << flagBits;
	}
	
	public long createArcCodePPP(FeatureTemplate.Arc temp, long x, long y, long z)
	{
		return ((((((x << tagNumBits) | y) << tagNumBits) | z ) << numArcFeatBits) | temp.ordinal()) << flagBits;
	}
	
	public long createArcCodePPPP(FeatureTemplate.Arc temp, long x, long y, long z, long w)
	{
		return ((((((((x << tagNumBits) | y) << tagNumBits) | z ) << tagNumBits) | w ) << numArcFeatBits) | temp.ordinal()) << flagBits;
	}
	
	public void addArcFeature(long code, FeatureVector fv)
	{
		int id = arcAlphabet.lookupIndex(code, numArcFeats);
		if (id >= 0) {
			fv.addEntry(id, 1.0);
			if (id == numArcFeats) ++numArcFeats;
		}
	}
	
}
