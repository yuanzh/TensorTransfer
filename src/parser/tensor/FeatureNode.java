package parser.tensor;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;
import parser.Options.TensorMode;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public abstract class FeatureNode {
	
	public DependencyInstance inst;
	public DependencyPipe pipe;
	public Options options;
	public ParameterNode pn;
	
	public static FeatureNode createFeatureNode(Options options, DependencyInstance inst, TensorTransfer model)
	{
		if (options.tensorMode == TensorMode.Threeway) {
			return new ThreewayFeatureNode(options, inst, model);
		}
		else if (options.tensorMode == TensorMode.Multiway) {
			return new MultiwayFeatureNode(options, inst, model);
		}
		else if (options.tensorMode == TensorMode.Hierarchical) {
			return new HierarchicalFeatureNode(options, inst, model);
		}
		else {
			Utils.ThrowException("not supported yet");
		}
		
		return null;
	}

	public abstract void initTabels();
	
	public abstract double getScore(int h, int m, int l);
	
	public abstract double addGradient(int h, int m, int l, double val, ParameterNode pn);
}

class FeatureDataItem {
	final FeatureVector fv;
	final double[] score;
	
	public FeatureDataItem(FeatureVector fv, double[] score)
	{
		this.fv = fv;
		this.score = score;
	}
}
