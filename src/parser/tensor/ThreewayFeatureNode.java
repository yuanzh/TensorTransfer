package parser.tensor;

import java.util.Arrays;

import parser.DependencyInstance;
import parser.Options;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public class ThreewayFeatureNode extends FeatureNode {
	public FeatureDataItem[] headData;
	public FeatureDataItem[] modData;
	public FeatureDataItem[] ddData;
	public FeatureDataItem[] labelData;
	
	public ThreewayFeatureNode(Options options, DependencyInstance inst, TensorTransfer model) {
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
		headData = new FeatureDataItem[n];
		modData = new FeatureDataItem[n];
		for (int i = 0; i < n; ++i) {
			ParameterNode hpn = pn.node[0];
			ParameterNode mpn = pn.node[1];
			FeatureVector fv = pipe.ff.createThreewayPosFeatures(inst, i, hpn.featureSize, hpn.featureBias);
			
			Utils.Assert(rank == mpn.rank);
			double[] headScore = new double[rank];
			double[] modScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				headScore[r] = fv.dotProduct(hpn.param[r]);
				modScore[r] = fv.dotProduct(mpn.param[r]);
			}
			
			headData[i] = new FeatureDataItem(fv, headScore);
			modData[i] = new FeatureDataItem(fv, modScore);
		}
		
		// direction and distance
		int d = ParameterNode.d;
		ddData = new FeatureDataItem[2 * d];
		ParameterNode dpn = pn.node[2];
		Utils.Assert(rank == dpn.rank);
		for (int i = 0; i < 2 * d; ++i) {
			FeatureVector fv = pipe.fr.getDDFv(i);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(dpn.param[r]);
			}
			ddData[i] = new FeatureDataItem(fv, score);
		}
		
		// label
		if (options.learnLabel) {
			int labelNum = pn.labelNum;
			labelData = new FeatureDataItem[labelNum];
			ParameterNode lpn = pn.node[3];
			Utils.Assert(rank == lpn.rank);
			for (int i = 0; i < labelNum; ++i) {
				FeatureVector fv = pipe.fr.getLabelFv(i);
				double[] score = new double[rank];
				for (int r = 0; r < rank; ++r) {
					score[r] = fv.dotProduct(lpn.param[r]);
				}
				labelData[i] = new FeatureDataItem(fv, score);
			}
		}
	}

	@Override
	public double getScore(int h, int m, int label) {
		double[] headScore = headData[h].score;
		double[] modScore = modData[m].score;
		double[] ddScore = ddData[pipe.ff.getBinnedDistance(h - m)].score;
		
		if (options.learnLabel) {
			double[] labelScore = labelData[label].score;
			return Utils.sum(Utils.dot(headScore, modScore, ddScore, labelScore));
		}
		else {
			return Utils.sum(Utils.dot(headScore, modScore, ddScore));
		}
	}
	
	@Override
	public double addGradient(int h, int m, int label, double val, ParameterNode pn) {
		// assume that dfv is already cleaned
		
		int binDist = pipe.ff.getBinnedDistance(h - m);
		double[] v = new double[pn.rank];
		Arrays.fill(v, val);
		
		double[] g = null; 
		
		// update h
		g = Utils.dot(v, modData[m].score, ddData[binDist].score);
		if (options.learnLabel)
			g = Utils.dot(g, labelData[label].score);
		ParameterNode hpn = pn.node[0];
		for (int r = 0; r < hpn.rank; ++r) {
			hpn.dFV[r].addEntries(headData[h].fv, g[r]);
		}
		
//		for (int r = 0; r < hpn.rank; ++r) {
//			System.out.printf("%.4f ", g[r]);
//		}
//		try { System.in.read(); } catch (IOException e) { e.printStackTrace(); }
		
		// update m
		g = Utils.dot(v, headData[h].score, ddData[binDist].score);
		if (options.learnLabel)
			g = Utils.dot(g, labelData[label].score);
		ParameterNode mpn = pn.node[1];
		for (int r = 0; r < mpn.rank; ++r) {
			mpn.dFV[r].addEntries(modData[m].fv, g[r]);
		}
		
		// update dd
		g = Utils.dot(v, headData[h].score, modData[m].score);
		if (options.learnLabel)
			g = Utils.dot(g, labelData[label].score);
		ParameterNode dpn = pn.node[2];
		for (int r = 0; r < dpn.rank; ++r) {
			dpn.dFV[r].addEntries(ddData[binDist].fv, g[r]);
		}
		
		// update label
		if (options.learnLabel) {
			g = Utils.dot(v, headData[h].score, modData[m].score, ddData[binDist].score);
			ParameterNode lpn = pn.node[3];
			for (int r = 0; r < lpn.rank; ++r) {
				lpn.dFV[r].addEntries(labelData[label].fv, g[r]);
			}
		}
		
		if (options.learnLabel) {
			double[] labelScore = labelData[label].score;
			return Utils.sum(Utils.dot(headData[h].score, modData[m].score, ddData[binDist].score, labelScore));
		}
		else {
			return Utils.sum(Utils.dot(headData[h].score, modData[m].score, ddData[binDist].score));
		}
	}
}
