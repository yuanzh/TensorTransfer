package parser.tensor;

import java.util.Arrays;

import parser.DependencyInstance;
import parser.Options;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public class MultiwayFeatureNode extends FeatureNode {

	public FeatureDataItem[] headContext;
	public FeatureDataItem[] modContext;
	public FeatureDataItem[] head;
	public FeatureDataItem[] mod;
	public FeatureDataItem[] dd;
	public FeatureDataItem[] label;
	
	public MultiwayFeatureNode(Options options, DependencyInstance inst, TensorTransfer model) {
		this.inst = inst;
		this.options = options;
		pipe = model.pipe;
		pn = model.parameters.pn;
	}
	
	@Override
	public void initTabels() {
		int n = inst.length;
		int rank = pn.node[0].rank;
		
		// POS
		headContext = new FeatureDataItem[n];
		modContext = new FeatureDataItem[n];
		head = new FeatureDataItem[n];
		mod = new FeatureDataItem[n];
		for (int i = 0; i < n; ++i) {
			int p = inst.postagids[i];
			int pp = i > 0 ? inst.postagids[i - 1] : pipe.ff.TOKEN_START;
			int np = i < n - 1 ? inst.postagids[i + 1] : pipe.ff.TOKEN_END;

			ParameterNode hpn = pn.node[0];
			ParameterNode mpn = pn.node[1];
			FeatureVector fv = pipe.fr.getPOSFv(p);
			Utils.Assert(rank == hpn.rank && rank == mpn.rank);
			double[] headScore = new double[rank];
			double[] modScore = new double[rank];
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			head[i] = new FeatureDataItem(fv, headScore);
			mod[i] = new FeatureDataItem(fv, modScore);

			hpn = pn.node[2];
			mpn = pn.node[3];
			fv = pipe.fr.getContextFv(pp, np);
			Utils.Assert(rank == hpn.rank && rank == mpn.rank);
			headScore = new double[rank];
			modScore = new double[rank];
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			headContext[i] = new FeatureDataItem(fv, headScore);
			modContext[i] = new FeatureDataItem(fv, modScore);
		}
		
		// direction and distance
		int d = ParameterNode.d;
		dd = new FeatureDataItem[2 * d];
		ParameterNode dpn = pn.node[4];
		Utils.Assert(rank == dpn.rank);
		for (int i = 0; i < 2 * d; ++i) {
			FeatureVector fv = pipe.fr.getDDFv(i);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(dpn.param[r]);
			}
			dd[i] = new FeatureDataItem(fv, score);
		}
		
		// label
		if (options.learnLabel) {
			int labelNum = pn.labelNum;
			label = new FeatureDataItem[labelNum];
			ParameterNode lpn = pn.node[5];
			Utils.Assert(rank == lpn.rank);
			for (int i = 0; i < labelNum; ++i) {
				FeatureVector fv = pipe.fr.getLabelFv(i);
				double[] score = new double[rank];
				for (int r = 0; r < rank; ++r) {
					score[r] = fv.dotProduct(lpn.param[r]);
				}
				label[i] = new FeatureDataItem(fv, score);
			}
		}
	}

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

	@Override
	public double addGradient(int h, int m, int l, double val, ParameterNode pn) {
		// assume that dfv is already cleaned
		
		double[] headScore = head[h].score;
		double[] modScore = mod[m].score;
		double[] hcScore = headContext[h].score;
		double[] mcScore = modContext[m].score;
		double[] ddScore = dd[pipe.ff.getBinnedDistance(h - m)].score;
		double[] lScore = options.learnLabel ? label[l].score : null;

		ParameterNode hpn = pn.node[0];
		ParameterNode mpn = pn.node[1];
		ParameterNode hcpn = pn.node[2];
		ParameterNode mcpn = pn.node[3];
		ParameterNode dpn = pn.node[4];
		ParameterNode lpn = options.learnLabel ? pn.node[5] : null;

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
		
		if (options.learnLabel) {
			double[] labelScore = label[l].score;
			return Utils.sum(Utils.dot(headScore, modScore, hcScore, mcScore, ddScore, labelScore));
		}
		else {
			return Utils.sum(Utils.dot(headScore, modScore, hcScore, mcScore, ddScore));
		}
	}

}
