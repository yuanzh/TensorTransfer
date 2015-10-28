package parser.tensor;

import java.util.Arrays;

import parser.DependencyInstance;
import parser.Options;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public class TMultiwayFeatureNode extends FeatureNode {

	public FeatureDataItem[] headLexicalData;
	public FeatureDataItem[] modLexicalData;
	public FeatureDataItem[] headContextData;
	public FeatureDataItem[] modContextData;
	public FeatureDataItem[] headData;
	public FeatureDataItem[] modData;
	public FeatureDataItem[] ddData;
	public FeatureDataItem[] labelData;
	
	public FeatureDataItem emptyLabelData;

	public TMultiwayFeatureNode(Options options, DependencyInstance inst, TensorTransfer model) {
		this.inst = inst;
		this.options = options;
		pipe = model.pipe;
		pn = model.parameters.pn;
	}
	
	@Override
	public void initTabels() {
		int n = inst.length;
		int rank = pn.rank;
		
		if (options.lexical) {
			headLexicalData = new FeatureDataItem[n];
			modLexicalData = new FeatureDataItem[n];
			for (int i = 0; i < n; ++i) {
				ParameterNode hpn = pn.node[7];
				ParameterNode mpn = pn.node[8];
				FeatureVector fv = pipe.ff.createLexicalFeatures(inst, i, hpn.featureSize, hpn.featureBias);
				double[] headScore = new double[rank];
				double[] modScore = new double[rank];
				
				for (int r = 0; r < rank; ++r) {
					headScore[r] = fv.dotProduct(hpn.param[r]);
					modScore[r] = fv.dotProduct(mpn.param[r]);
				}
				headLexicalData[i] = new FeatureDataItem(fv, headScore);
				modLexicalData[i] = new FeatureDataItem(fv, modScore);
			}
		}
		
		headContextData = new FeatureDataItem[n];
		modContextData = new FeatureDataItem[n];
		headData = new FeatureDataItem[n];
		modData = new FeatureDataItem[n];
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i > 0 ? inst.postagids[i - 1] : pipe.ff.TOKEN_START;
			int np = i < n - 1 ? inst.postagids[i + 1] : pipe.ff.TOKEN_END;

			// context
			ParameterNode hpn = pn.node[2];
			ParameterNode mpn = pn.node[3];
			FeatureVector fv = pipe.fr.getContextFv(pp, np, inst.lang);
			
			Utils.Assert(hpn.rank == rank && mpn.rank == rank);
			double[] headScore = new double[rank];
			double[] modScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			
			headContextData[i] = new FeatureDataItem(fv, headScore);
			modContextData[i] = new FeatureDataItem(fv, modScore);
			
			// pos
			hpn = pn.node[0];
			mpn = pn.node[1];
			fv = pipe.fr.getPOSFv(p);
			
			Utils.Assert(hpn.rank == rank && mpn.rank == rank);
			headScore = new double[rank];
			modScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			
			headData[i] = new FeatureDataItem(fv, headScore);
			modData[i] = new FeatureDataItem(fv, modScore);
		}
		
		// direction and distance and typo
		int d = ParameterNode.d;
		ddData = new FeatureDataItem[2 * d];
		ParameterNode dpn = pn.node[5];
		Utils.Assert(rank == dpn.rank);
		for (int i = 0; i < 2 * d; ++i) {
			FeatureVector fv = pipe.fr.getDDTypoFv(i, inst.lang);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(dpn.param[r]);
			}
			ddData[i] = new FeatureDataItem(fv, score);
		}
		
		// label
		if (options.learnLabel) {
			ParameterNode lpn = pn.node[6];
			Utils.Assert(rank == lpn.rank);
			int labelNum = pn.labelNum;
			labelData = new FeatureDataItem[labelNum];
			for (int i = 0; i < labelNum; ++i) {
				FeatureVector fv = pipe.fr.getLabelFv(i);
				double[] score = new double[rank];
				for (int r = 0; r < rank; ++r) {
					score[r] = fv.dotProduct(lpn.param[r]);
				}
				labelData[i] = new FeatureDataItem(fv, score);
			}
			
			// empty label
			FeatureVector fv = pipe.fr.getLabelFv(-1);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(lpn.param[r]);
			}
			emptyLabelData = new FeatureDataItem(fv, score);
		}
	}

	@Override
	public double getScore(int h, int m, int label) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] lexScore = null;
		if (options.lexical) {
			int rank = pn.rank;
			lexScore = new double[rank];
			
			double[] headLexicalScore = headLexicalData[h].score;
			double[] modLexicalScore = modLexicalData[m].score;
			for (int r = 0; r < rank; ++r) {
				lexScore[r] += headLexicalScore[r] * modLexicalScore[r];
			}
		}

		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		double[] lScore = label < 0 ? emptyLabelData.score : labelData[label].score;
		
		ParameterNode tpn = pn.node[4];
		double[] tScore = new double[tpn.rank];
		FeatureVector fv = pipe.ff.createAllTypoFeatures(hp, mp, label, binDist, lang, tpn.featureSize, tpn.featureBias);
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += fv.dotProduct(tpn.param[r]);
		}
		
		if (options.lexical) 
			return Utils.sum(Utils.dot(lexScore, hcScore, mcScore, tScore, hScore, mScore, ddScore, lScore));
		else
			return Utils.sum(Utils.dot(hcScore, mcScore, tScore, hScore, mScore, ddScore, lScore));
	}

	@Override
	public double addGradient(int h, int m, int label, double val,
			ParameterNode pn) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] v = new double[pn.rank];
		Arrays.fill(v, val);

		ParameterNode hcpn = pn.node[2];
		ParameterNode mcpn = pn.node[3];
		ParameterNode lpn = pn.node[6];
		ParameterNode tpn = pn.node[4];
		ParameterNode hpn = pn.node[0];
		ParameterNode mpn = pn.node[1];
		ParameterNode dpn = pn.node[5];
		
		ParameterNode hlpn = options.lexical ? pn.node[7] : null;
		ParameterNode mlpn = options.lexical ? pn.node[8] : null;

		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		double[] lScore = label < 0 ? emptyLabelData.score : labelData[label].score;
		
		double[] tScore = new double[tpn.rank];
		FeatureVector tfv = pipe.ff.createAllTypoFeatures(hp, mp, label, binDist, lang, tpn.featureSize, tpn.featureBias);
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += tfv.dotProduct(tpn.param[r]);
		}

		double[] lexScore = null;
		if (options.lexical) {
			int rank = pn.rank;
			lexScore = new double[rank];
			
			double[] headLexicalScore = headLexicalData[h].score;
			double[] modLexicalScore = modLexicalData[m].score;
			for (int r = 0; r < rank; ++r) {
				lexScore[r] += headLexicalScore[r] * modLexicalScore[r];
			}
		
			for (int r = 0; r < rank; ++r) {
				double tmp = v[r] * hcScore[r] * mcScore[r] * hScore[r] * mScore[r] * ddScore[r] * lScore[r] * tScore[r];
				// update head lexical
				double dv = tmp * modLexicalScore[r];
				hlpn.dFV[r].addEntries(headLexicalData[h].fv, dv);

				// update mod lexical
				dv = tmp * headLexicalScore[r];
				mlpn.dFV[r].addEntries(modLexicalData[m].fv, dv);
			}
		}

		// head/mod context
		int rank = pn.rank;
		double[] g = new double[rank];
		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, mcScore, hScore, mScore, ddScore, lScore, tScore);
		else
			g = Utils.dot_s(g, v, mcScore, hScore, mScore, ddScore, lScore, tScore);
		for (int r = 0; r < hcpn.rank; ++r) {
			hcpn.dFV[r].addEntries(headContextData[h].fv, g[r]);
		}

		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, hcScore, hScore, mScore, ddScore, lScore, tScore);
		else
			g = Utils.dot_s(g, v, hcScore, hScore, mScore, ddScore, lScore, tScore);
		for (int r = 0; r < mcpn.rank; ++r) {
			mcpn.dFV[r].addEntries(modContextData[m].fv, g[r]);
		}
		
		// head/mod
		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, mcScore, hcScore, mScore, ddScore, lScore, tScore);
		else
			g = Utils.dot_s(g, v, mcScore, hcScore, mScore, ddScore, lScore, tScore);
		for (int r = 0; r < hpn.rank; ++r) {
			hpn.dFV[r].addEntries(headData[h].fv, g[r]);
		}

		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, hcScore, hScore, mcScore, ddScore, lScore, tScore);
		else
			g = Utils.dot_s(g, v, hcScore, hScore, mcScore, ddScore, lScore, tScore);
		for (int r = 0; r < mpn.rank; ++r) {
			mpn.dFV[r].addEntries(modData[m].fv, g[r]);
		}
		
		// label
		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, hcScore, mcScore, hScore, mScore, ddScore, tScore);
		else
			g = Utils.dot_s(g, v, hcScore, mcScore, hScore, mScore, ddScore, tScore);
		for (int r = 0; r < lpn.rank; ++r) {
			FeatureVector fv = label < 0 ? emptyLabelData.fv : labelData[label].fv;
			lpn.dFV[r].addEntries(fv, g[r]);
		}
		
		// dd
		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, hcScore, mcScore, hScore, mScore, lScore, tScore);
		else
			g = Utils.dot_s(g, v, hcScore, mcScore, hScore, mScore, lScore, tScore);
		for (int r = 0; r < dpn.rank; ++r) {
			dpn.dFV[r].addEntries(ddData[binDist].fv, g[r]);
		}
		
		// typo
		if (options.lexical)
			g = Utils.dot_s(g, v, lexScore, hcScore, mcScore, hScore, mScore, lScore, ddScore);
		else
			g = Utils.dot_s(g, v, hcScore, mcScore, hScore, mScore, lScore, ddScore);
		for (int r = 0; r < tpn.rank; ++r) {
			tpn.dFV[r].addEntries(tfv, g[r]);
		}

		if (options.lexical) 
			return Utils.sum(Utils.dot(lexScore, hcScore, mcScore, tScore, hScore, mScore, ddScore, lScore));
		else
			return Utils.sum(Utils.dot(hcScore, mcScore, tScore, hScore, mScore, ddScore, lScore));
	}

}
