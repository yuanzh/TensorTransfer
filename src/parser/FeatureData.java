package parser;

import java.util.Arrays;

import parser.Options.FeatureMode;
import parser.decoding.DependencyDecoder;
import parser.tensor.FeatureNode;
import utils.FeatureVector;
import utils.Utils;

public class FeatureData {
	static double NULL = Double.NEGATIVE_INFINITY;
	
	DependencyInstance inst;
	DependencyPipe pipe;
	Options options;
	Parameters parameters;
	
	TensorTransfer pruner;
	DependencyDecoder prunerDecoder;
	
	final int len;					// sentence length
	final int ntypes;				// number of label types
	final double gamma;
	final boolean addLoss;
	final FeatureNode fn;
	
	int numarcs;					// number of un-pruned arcs and gold arcs (if indexGoldArcs == true)
	int[] arc2id;					// map (h->m) arc to an id in [0, numarcs-1]
	boolean[] isPruned;				// whether a (h->m) arc is pruned
	int[] edges, st;
	int numedges;						// number of un-pruned arcs

	private FeatureVector[] arcFvs;			// 1st order arc feature vectors, no label
	private double[] arcLabelScores;		// 1st order arc scores (including tensor, and label)
	private int[] bestLabel;				// assume label score is first order, so we can pre-compute the best label for each arc

	public FeatureData(DependencyInstance inst, TensorTransfer model, boolean indexGoldArcs, boolean addLoss) 
	{
		this.inst = inst;
		pipe = model.pipe;
		options = model.options;
		parameters = model.parameters;
		this.addLoss = addLoss;
			
		Utils.Assert(pruner == null || pruner.options.featureMode == FeatureMode.Basic);
		
		fn = FeatureNode.createFeatureNode(options, inst, model);
		fn.initTabels();
		
		len = inst.length;
		ntypes = pipe.types.length;
		gamma = options.gamma;
		Utils.Assert(ntypes == parameters.pn.labelNum 
				&& gamma == parameters.pn.gamma
				&& gamma == parameters.gamma);
		
		arcFvs = new FeatureVector[len * len];
		arcLabelScores = new double[len * len];
		Arrays.fill(arcLabelScores, NULL);
		if (options.learnLabel) {
			bestLabel = new int[len * len];
			Arrays.fill(bestLabel, -1);
		}

		// calculate 1st order feature vectors and scores
		initFirstOrderTables();
	}
	private void initFirstOrderTables() 
	{
		boolean nopruning = !options.pruning || pruner == null || options.featureMode == FeatureMode.Basic;
		
		for (int h = 0; h < len; ++h)
			for (int m = 1; m < len; ++m) 
				if (h != m && (nopruning || arc2id[h * len + m] != -1)) {
					arcFvs[h * len + m] = pipe.ff.createArcFeatures(inst, h, m);
					double score = parameters.getScore(arcFvs[h * len + m]) * gamma;
					
					if (options.learnLabel) {
						int optLabel = -1;
						double optScore = NULL;
						for (int l = 0; l < ntypes; ++l) {
							FeatureVector lfv = pipe.ff.createArcLabelFeatures(inst, h, m, l);
							double lScore = parameters.getLabelScore(lfv) * gamma;
							double tScore = gamma < 1.0 ? fn.getScore(h, m, l) * (1-gamma) : 0.0;
							//System.out.println(tScore + "\t" + h + "\t" + m + "\t" + l);
							double loss = getLoss(h, l, inst.heads[m], inst.deplbids[m]);
							if (score + lScore + tScore + loss > optScore + 1e-10) {
								optScore = score + lScore + tScore + loss;
								optLabel = l;
							}
						}
						arcLabelScores[h * len + m] = optScore;
						bestLabel[h * len + m] = optLabel;
						//System.out.println(h + "\t" + m + "\t" + optScore + "\t" + optLabel);
						//Utils.block();
					}
					else {
						double tScore = gamma < 1.0 ? fn.getScore(h, m, -1) * (1-gamma) : 0.0;
						double loss = getLoss(h, -1, inst.heads[m], -1);
						arcLabelScores[h * len + m] = score + tScore + loss;
					}
				}
	}
	
	public double getLoss(int h, int l, int gh, int gl) {
		if (!addLoss)
			return 0.0;
		if (h != gh)
			return 1.0;
		else if (l != gl)
			return 0.5;
		else
			return 0.0;
	}

	public int getBestLabel(int h, int m) {
		return bestLabel[h * len + m];
	}
	
	public double getArcScoreWithLoss(int h, int m) {
		return arcLabelScores[h * len + m];
	}
	
	public double getArcScoreWithoutLoss(int h, int m, int l) {
		double score = parameters.getScore(arcFvs[h * len + m]);
		double labelScore = !options.learnLabel ? 0.0
				: parameters.getLabelScore(pipe.ff.createArcLabelFeatures(inst, h, m, l));
		double tensorScore = gamma < 1.0 ? fn.getScore(h, m, l) : 0.0;
		return (score + labelScore) * gamma + tensorScore * (1 - gamma);
	}
	
	public FeatureVector getFeatureVector(DependencyInstance inst) {
		FeatureVector fv = new FeatureVector(pipe.ff.numArcFeats);
		int n = inst.length;
		
		for (int i = 1; i < n; ++i) {
			int h = inst.heads[i];
			fv.addEntries(arcFvs[h * n + i]);
			fv.addEntries(pipe.ff.createArcLabelFeatures(inst, h, i, inst.deplbids[i]));
		}
		
		return fv;
	}

	public FeatureVector getFeatureDifference(DependencyInstance gold, 
			DependencyInstance pred)
	{
		FeatureVector dfv = getFeatureVector(gold);
		dfv.addEntries(getFeatureVector(pred), -1.0);

		return dfv;
	}
}