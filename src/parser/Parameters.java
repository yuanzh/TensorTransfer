package parser;

import java.io.Serializable;
import java.util.Arrays;

import parser.Options.UpdateMode;
import parser.tensor.ParameterNode;
import utils.FeatureVector;
import utils.Utils;

public class Parameters implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public ParameterNode pn;
	public Options options;
	public DependencyPipe pipe;
	
	public double gamma;
	
	public double adaAlpha, adaEps, C;
	
	public double[] params;
	public int size;
	public transient double[] back, total, sg;
	
	public Parameters(Options options, DependencyPipe pipe, ParameterNode pn) 
	{
		this.options = options;
		this.pipe = pipe;
		this.pn = pn;
		
		gamma = options.gamma;
		
		size = pipe.ff.numArcFeats;		
		params = new double[size];
		
		if (options.updateMode == UpdateMode.AdaGrad) {
			sg = new double[size];
			adaAlpha = options.AdaAlpha;
			adaEps = options.AdaEps;
		}
		else if (options.updateMode == UpdateMode.MIRA) {
			total = new double[size];
			C = options.MIRAC;
		}
	}

	public double getScore(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}

	public double getLabelScore(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}
	
	public double update(DependencyInstance gold, DependencyInstance pred,
			FeatureData fd, int updCnt)
	{
		int n = gold.length;
		
		double loss = 0.0;
		for (int i = 1; i < n; ++i)
			loss += fd.getLoss(pred.heads[i], pred.deplbids[i], gold.heads[i], gold.deplbids[i]);
		
		double loss2 = loss;
		// compute loss2
		for (int i = 1; i < n; ++i) {
			loss2 -= fd.getArcScoreWithoutLoss(gold.heads[i], i, gold.deplbids[i]);
			loss2 += fd.getArcScoreWithoutLoss(pred.heads[i], i, pred.deplbids[i]);
		}

		// tensor
		if (gamma < 1.0) {
			for (int i = 1; i < n; ++i) {
				loss -= fd.fn.addGradient(gold.heads[i], i, gold.deplbids[i], 1.0, pn) * (1 - gamma);
				loss += fd.fn.addGradient(pred.heads[i], i, pred.deplbids[i], -1.0, pn) * (1 - gamma);
				if (options.learnLabel)
					Utils.Assert(pred.deplbids[i] == fd.getBestLabel(pred.heads[i], i));
				//System.out.println(pred.heads[i] + "\t" + gold.heads[i] + "\t" + pred.deplbids[i] + "\t" + gold.deplbids[i]);
			}
		}
		//System.out.println(loss);
		//Utils.block();
		
		// traditional features
    	FeatureVector dt = fd.getFeatureDifference(gold, pred);
    	loss -= dt.dotProduct(params) * gamma;
    	
    	Utils.Assert(Math.abs(loss - loss2) < 1e-6);
		
		// update
		if (loss > 0) {
			switch (options.updateMode) {
			case AdaGrad:
				for (int i = 0, L = dt.size(); i < L; ++i) {
					int x = dt.x(i);
					double g = dt.value(i) * gamma;
					sg[x] += g * g;
					params[x] += adaAlpha / Math.sqrt(sg[x] + adaEps) * g;
				}
				
				if (gamma < 1.0)
					pn.updateAda();
				
				break;
			case MIRA:
				double l2norm = gamma < 1.0 ? pn.gradientl2Norm() * (1 - gamma) * (1 - gamma) : 0.0;
				l2norm += dt.Squaredl2NormUnsafe() * gamma * gamma;
				double alpha = loss / l2norm;
		        //System.out.println(alpha + " " + l2norm);
		        //Utils.block();
				
				double lr = Math.min(alpha, C);
				for (int i = 0, L = dt.size(); i < L; ++i) {
					int x = dt.x(i);
					double g = dt.value(i) * gamma;
					params[x] += lr * g;
					total[x] += lr * updCnt * g;
				}
				
				if (gamma < 1.0)
					pn.updateMIRA(alpha, updCnt);
		    		
				break;
			default:
				break;
			}
		}
		
		return loss;
	}	
	
	public void averageParameters(int T) {
		back = params;
		double[] avgParams = new double[size];
		for (int i = 0; i < size; ++i) {
			avgParams[i] = (params[i] * (T+1) - total[i])/T;			
		}		
		params = avgParams;

		pn.averageParameters(T);
	}
	
	public void unaverageParameters() {
		params = back;
		pn.unaverageParameters();
	}
}
