package parser.tensor;

import java.util.Arrays;

import parser.DependencyInstance;
import parser.Options;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public class HierarchicalFeatureNode extends FeatureNode {
	public FeatureDataItem[] headLexical;
	public FeatureDataItem[] modLexical;
	public FeatureDataItem[] headContext;
	public FeatureDataItem[] modContext;
	public FeatureDataItem[] label;
	public FeatureDataItem[] head;
	public FeatureDataItem[] mod;
	public FeatureDataItem[] dd;
	
	double[] typoScore;
	double[] arcScore;
	double[] finalScore;
	double[] gScore;
	double[] g2Score;
	double[] g3Score;
	
	public HierarchicalFeatureNode(Options options, DependencyInstance inst, TensorTransfer model) {
		this.inst = inst;
		this.options = options;
		pipe = model.pipe;
		pn = model.parameters.pn;
	}
	
	@Override
	public void initTabels() {
		int n = inst.length;
		int rank = pn.rank;
		ParameterNode delexical = pn;
		
		if (options.lexical) {
			// TODO: add lexical
			delexical = pn.node[1];
		}
		
		headContext = new FeatureDataItem[n];
		modContext = new FeatureDataItem[n];
		head = new FeatureDataItem[n];
		mod = new FeatureDataItem[n];
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i > 0 ? inst.postagids[i - 1] : pipe.ff.TOKEN_START;
			int np = i < n - 1 ? inst.postagids[i + 1] : pipe.ff.TOKEN_END;

			// context
			ParameterNode hpn = delexical.node[0];
			ParameterNode mpn = delexical.node[1];
			FeatureVector fv = pipe.fr.getContextFv(pp, np, inst.lang);
			
			Utils.Assert(hpn.rank == rank && mpn.rank == rank);
			double[] headScore = new double[rank];
			double[] modScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			
			headContext[i] = new FeatureDataItem(fv, headScore);
			modContext[i] = new FeatureDataItem(fv, modScore);
			
			// pos
			hpn = options.learnLabel ? delexical.node[2].node[1].node[0] 
					: delexical.node[2].node[0];
			mpn = options.learnLabel ? delexical.node[2].node[1].node[1] 
					: delexical.node[2].node[1];
			fv = pipe.fr.getPOSFv(p);
			
			Utils.Assert(hpn.rank == rank && mpn.rank == rank);
			headScore = new double[rank];
			modScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			
			head[i] = new FeatureDataItem(fv, headScore);
			mod[i] = new FeatureDataItem(fv, modScore);
		}
		
		// direction and distance and typo
		int d = ParameterNode.d;
		dd = new FeatureDataItem[2 * d];
		ParameterNode dpn = options.learnLabel ? delexical.node[2].node[1].node[2] 
				: delexical.node[2].node[2];
		Utils.Assert(rank == dpn.rank);
		for (int i = 0; i < 2 * d; ++i) {
			FeatureVector fv = pipe.fr.getDDTypoFv(i, inst.lang);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(dpn.param[r]);
			}
			dd[i] = new FeatureDataItem(fv, score);
		}
		
		// label
		if (options.learnLabel) {
			ParameterNode lpn = delexical.node[2].node[0];
			Utils.Assert(rank == lpn.rank);
			int labelNum = pn.labelNum;
			label = new FeatureDataItem[labelNum];
			for (int i = 0; i < labelNum; ++i) {
				FeatureVector fv = pipe.fr.getLabelFv(i);
				double[] score = new double[rank];
				for (int r = 0; r < rank; ++r) {
					score[r] = fv.dotProduct(lpn.param[r]);
				}
				label[i] = new FeatureDataItem(fv, score);
			}
		}
		
		// temporary array
		typoScore = new double[rank];
		arcScore = new double[rank];
		finalScore = new double[rank];
		gScore = new double[rank];
		g2Score = new double[rank];
		g3Score = new double[rank];
	}

	
	@Override
	public double getScore(int h, int m, int l) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] hcScore = headContext[h].score;
		double[] mcScore = modContext[m].score;
		
		double[] hScore = head[h].score;
		double[] mScore = mod[m].score;
		double[] ddScore = dd[binDist].score;
		//double[] tScore = Utils.dot(hScore, mScore, ddScore);
		double[] tScore = Utils.dot_s(typoScore, hScore, mScore, ddScore);
		
		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		FeatureVector fv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		ParameterNode tpn = options.learnLabel ? delexical.node[2].node[1] : delexical.node[2];
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += fv.dotProduct(tpn.param[r]);
		}
		
		if (options.learnLabel) {
			double[] lScore = label[l].score;
			//double[] arcScore = Utils.dot(lScore, tScore);
			double[] aScore = Utils.dot_s(arcScore, lScore, tScore);
			fv = pipe.fr.getSVOFv(hp, mp, l, binDist, lang);
			ParameterNode apn = delexical.node[2];
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] += fv.dotProduct(apn.param[r]);
			tScore = aScore;
		}
		
		//return Utils.sum(Utils.dot(hcScore, mcScore, tScore));
		return Utils.sum(Utils.dot_s(finalScore, hcScore, mcScore, tScore));
	}
	
	/*
	@Override
	public double getScore(int h, int m, int l) {
		double[] headScore = head[h].score;
		double[] modScore = mod[m].score;
		double[] hcScore = headContext[h].score;
		double[] mcScore = modContext[m].score;
		double[] ddScore = dd[pipe.ff.getBinnedDistance(h - m)].score;
		
		if (options.learnLabel) {
			double[] labelScore = label[l].score;
			return Utils.sum(Utils.dot(headScore, modScore, hcScore, mcScore, ddScore, labelScore));
		}
		else {
			return Utils.sum(Utils.dot(headScore, modScore, hcScore, mcScore, ddScore));
		}
	}
*/

	@Override
	public double addGradient(int h, int m, int l, double val, ParameterNode pn) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		ParameterNode hcpn = delexical.node[0];
		ParameterNode mcpn = delexical.node[1];
		ParameterNode apn = options.learnLabel ? delexical.node[2] : null;
		ParameterNode lpn = options.learnLabel ? apn.node[0] : null;
		ParameterNode tpn = options.learnLabel ? apn.node[1] : delexical.node[2];
		ParameterNode hpn = tpn.node[0];
		ParameterNode mpn = tpn.node[1];
		ParameterNode dpn = tpn.node[2];
		
		double[] v = new double[pn.rank];
		Arrays.fill(v, val);
		
		double[] g = null; 

		double[] hcScore = headContext[h].score;
		double[] mcScore = modContext[m].score;
		
		double[] hScore = head[h].score;
		double[] mScore = mod[m].score;
		double[] ddScore = dd[binDist].score;
		//double[] tScore = Utils.dot(hScore, mScore, ddScore);
		double[] tScore = Utils.dot_s(typoScore, hScore, mScore, ddScore);
		
		FeatureVector tfv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += tfv.dotProduct(tpn.param[r]);
		}
		
		if (options.learnLabel) {
			double[] lScore = label[l].score;
			//double[] aScore = Utils.dot(lScore, tScore);
			double[] aScore = Utils.dot_s(arcScore, lScore, tScore);
			FeatureVector afv = pipe.fr.getSVOFv(hp, mp, l, binDist, lang);
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] += afv.dotProduct(apn.param[r]);
			
			// update head context
			//g = Utils.dot(v, mcScore, aScore);
			g = Utils.dot_s(gScore, v, mcScore, aScore);
			for (int r = 0; r < hcpn.rank; ++r) {
				hcpn.dFV[r].addEntries(headContext[h].fv, g[r]);
			}
			
			// update mod context
			//g = Utils.dot(v, hcScore, aScore);
			g = Utils.dot_s(gScore, v, hcScore, aScore);
			for (int r = 0; r < mcpn.rank; ++r) {
				mcpn.dFV[r].addEntries(modContext[m].fv, g[r]);
			}
			
			// update svo
			//g = Utils.dot(v, hcScore, mcScore);
			g = Utils.dot_s(gScore, v, hcScore, mcScore);
			for (int r = 0; r < apn.rank; ++r) {
				apn.dFV[r].addEntries(afv, g[r]);
			}
			
			// update label
			//double[] g2 = Utils.dot(g, tScore);
			double[] g2 = Utils.dot_s(g2Score, g, tScore);
			for (int r = 0; r < lpn.rank; ++r) {
				lpn.dFV[r].addEntries(label[l].fv, g2[r]);
			}
			
			// update typo
			//g2 = Utils.dot(g, lScore);
			g2 = Utils.dot_s(g2Score, g, lScore);
			for (int r = 0; r < tpn.rank; ++r) {
				tpn.dFV[r].addEntries(tfv, g2[r]);
			}
			
			// update head
			//double[] g3 = Utils.dot(g2, mScore, ddScore);
			double[] g3 = Utils.dot_s(g3Score, g2, mScore, ddScore);
			for (int r = 0; r < hpn.rank; ++r) {
				hpn.dFV[r].addEntries(head[h].fv, g3[r]);
			}
			
			// update mod
			//g3 = Utils.dot(g2, hScore, ddScore);
			g3 = Utils.dot_s(g3Score, g2, hScore, ddScore);
			for (int r = 0; r < mpn.rank; ++r) {
				mpn.dFV[r].addEntries(mod[m].fv, g3[r]);
			}
			
			// update dd
			//g3 = Utils.dot(g2, hScore, mScore);
			g3 = Utils.dot_s(g3Score, g2, hScore, mScore);
			for (int r = 0; r < dpn.rank; ++r) {
				dpn.dFV[r].addEntries(dd[binDist].fv, g3[r]);
			}
			
			return Utils.sum(Utils.dot_s(finalScore, hcScore, mcScore, aScore));
		}
		else {
			// update head context
			g = Utils.dot(v, mcScore, tScore);
			for (int r = 0; r < hcpn.rank; ++r) {
				hcpn.dFV[r].addEntries(headContext[h].fv, g[r]);
			}
			
			// update mod context
			g = Utils.dot(v, hcScore, tScore);
			for (int r = 0; r < mcpn.rank; ++r) {
				mcpn.dFV[r].addEntries(modContext[m].fv, g[r]);
			}
			
			// update typo
			g = Utils.dot(v, hcScore, mcScore);
			for (int r = 0; r < tpn.rank; ++r) {
				tpn.dFV[r].addEntries(tfv, g[r]);
			}
			
			// update head
			double[] g2 = Utils.dot(g, mScore, ddScore);
			for (int r = 0; r < hpn.rank; ++r) {
				hpn.dFV[r].addEntries(head[h].fv, g2[r]);
			}
			
			// update mod
			g2 = Utils.dot(g, hScore, ddScore);
			for (int r = 0; r < mpn.rank; ++r) {
				mpn.dFV[r].addEntries(mod[m].fv, g2[r]);
			}
			
			// update dd
			g2 = Utils.dot(g, hScore, mScore);
			for (int r = 0; r < dpn.rank; ++r) {
				dpn.dFV[r].addEntries(dd[binDist].fv, g2[r]);
			}
			
			return Utils.sum(Utils.dot_s(finalScore, hcScore, mcScore, tScore));
		}
	}

/*	
	@Override
	public void addGradient(int h, int m, int l, double val, ParameterNode pn) {
		// assume that dfv is already cleaned
		
		double[] headScore = head[h].score;
		double[] modScore = mod[m].score;
		double[] hcScore = headContext[h].score;
		double[] mcScore = modContext[m].score;
		double[] ddScore = dd[pipe.ff.getBinnedDistance(h - m)].score;
		double[] lScore = options.learnLabel ? label[l].score : null;

		ParameterNode hpn = pn.node[2].node[1].node[0];
		ParameterNode mpn = pn.node[2].node[1].node[1];
		ParameterNode hcpn = pn.node[0];
		ParameterNode mcpn = pn.node[1];
		ParameterNode dpn = pn.node[2].node[1].node[2];
		ParameterNode lpn = options.learnLabel ? pn.node[2].node[0] : null;

		int binDist = pipe.ff.getBinnedDistance(h - m);
		double[] v = new double[pn.rank];
		Arrays.fill(v, val);
		
		double[] g = null; 
		
		// update h
		g = Utils.dot(v, modScore, hcScore, mcScore, ddScore);
		if (options.learnLabel)
			g = Utils.dot(g, lScore);
		for (int r = 0; r < hpn.rank; ++r) {
			hpn.dFV[r].addEntries(head[h].fv, g[r]);
		}
		
		// update m
		g = Utils.dot(v, headScore, hcScore, mcScore, ddScore);
		if (options.learnLabel)
			g = Utils.dot(g, lScore);
		for (int r = 0; r < mpn.rank; ++r) {
			mpn.dFV[r].addEntries(mod[m].fv, g[r]);
		}
		
		// update hc
		g = Utils.dot(v, mcScore, headScore, modScore, ddScore);
		if (options.learnLabel)
			g = Utils.dot(g, lScore);
		for (int r = 0; r < hcpn.rank; ++r) {
			hcpn.dFV[r].addEntries(headContext[h].fv, g[r]);
		}
		
		// update mc
		g = Utils.dot(v, hcScore, headScore, modScore, ddScore);
		if (options.learnLabel)
			g = Utils.dot(g, lScore);
		for (int r = 0; r < mcpn.rank; ++r) {
			mcpn.dFV[r].addEntries(modContext[m].fv, g[r]);
		}
		
		// update dd
		g = Utils.dot(v, headScore, modScore, hcScore, mcScore);
		if (options.learnLabel)
			g = Utils.dot(g, lScore);
		for (int r = 0; r < dpn.rank; ++r) {
			dpn.dFV[r].addEntries(dd[binDist].fv, g[r]);
		}
		
		// update label
		if (options.learnLabel) {
			g = Utils.dot(v, headScore, modScore, hcScore, mcScore, ddScore);
			for (int r = 0; r < lpn.rank; ++r) {
				lpn.dFV[r].addEntries(label[l].fv, g[r]);
			}
		}
	}
*/
}
