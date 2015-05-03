package parser;

import gnu.trove.set.hash.TLongHashSet;

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
	public transient double[] dFV;
	public transient double totalLoss;
	
	public int updCnt;
	
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
		
		updCnt = 0;
		dFV = new double[size];
		totalLoss = 0.0;
	}

	public double getScore(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}

	public double getLabelScore(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}
	
	public double addGradient(DependencyInstance gold, DependencyInstance pred,
			FeatureData fd) {
		int n = gold.length;
		
		double loss = 0.0;
		for (int i = 1; i < n; ++i)
			loss += fd.getLoss(pred.heads[i], pred.deplbids[i], gold.heads[i], gold.deplbids[i]);
		
		// tensor
		if (gamma < 1.0) {
			for (int i = 1; i < n; ++i) {
				loss -= fd.fn.addGradient(gold.heads[i], i, gold.deplbids[i], 1.0, pn) * (1 - gamma);
				loss += fd.fn.addGradient(pred.heads[i], i, pred.deplbids[i], -1.0, pn) * (1 - gamma);
				if (options.learnLabel)
					Utils.Assert(pred.deplbids[i] == fd.getBestLabel(pred.heads[i], i));
			}
		}
		
		// traditional features
		FeatureVector dt = null;
		if (gamma > 0.0) {
			dt = fd.getFeatureDifference(gold, pred);
			loss -= dt.dotProduct(params) * gamma;
			for (int i = 0, L = dt.size(); i < L; ++i) {
				dFV[dt.x(i)] += dt.value(i);
			}
		}
		
		totalLoss += loss;
		
		return loss;
	}
	
	public void batchUpdate() {
		updCnt++;
		
		switch (options.updateMode) {
		case AdaGrad:
			if (gamma > 0.0) {
				for (int i = 0; i < size; ++i) {
					double g = (dFV[i] - options.lambda * params[i]) * gamma;
					sg[i] += g * g;
					params[i] += adaAlpha / Math.sqrt(sg[i] + adaEps) * g;
					dFV[i] = 0.0;
				}
			}
			
			if (gamma < 1.0)
				pn.batchUpdateAda();
			
			break;
		case MIRA:
			double l2norm = gamma < 1.0 ? pn.gradientl2Norm() * (1 - gamma) * (1 - gamma) : 0.0;
			if (gamma > 0.0) {
				for (int i = 0; i < size; ++i) {
					double g = (dFV[i] - options.lambda * params[i]) * gamma;
					l2norm += g * g;
				}
			}
			double alpha = totalLoss / l2norm;
			
			if (gamma > 0.0) {
				double lr = Math.min(alpha, C);
				for (int i = 0; i < size; ++i) {
					double g = (dFV[i] - options.lambda * params[i]) * gamma;
					params[i] += lr * g;
					total[i] += lr * updCnt * g;
					dFV[i] = 0.0;
				}
			}
			
			if (gamma < 1.0)
				pn.updateMIRA(alpha, updCnt);
	    		
			break;
		default:
			break;
		}
		
		totalLoss = 0.0;
	}
	
	public double update(DependencyInstance gold, DependencyInstance pred,
			FeatureData fd)
	{
		updCnt++;
		int n = gold.length;
		
		double loss = 0.0;
		for (int i = 1; i < n; ++i)
			loss += fd.getLoss(pred.heads[i], pred.deplbids[i], gold.heads[i], gold.deplbids[i]);
		
		//double loss2 = loss;
		// compute loss2
		//for (int i = 1; i < n; ++i) {
		//	loss2 -= fd.getArcScoreWithoutLoss(gold.heads[i], i, gold.deplbids[i]);
		//	loss2 += fd.getArcScoreWithoutLoss(pred.heads[i], i, pred.deplbids[i]);
		//}

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
		FeatureVector dt = null;
		if (gamma > 0.0) {
			dt = fd.getFeatureDifference(gold, pred);
			loss -= dt.dotProduct(params) * gamma;
		}
    	
    	//Utils.Assert(Math.abs(loss - loss2) < 1e-6);
		
		// update
		if (loss > 0) {
			switch (options.updateMode) {
			case AdaGrad:
				if (gamma > 0.0) {
					for (int i = 0, L = dt.size(); i < L; ++i) {
						int x = dt.x(i);
						double g = dt.value(i) * gamma;
						sg[x] += g * g;
						params[x] += adaAlpha / Math.sqrt(sg[x] + adaEps) * g;
					}
				}
				
				if (gamma < 1.0)
					pn.updateAda();
				
				break;
			case MIRA:
				double l2norm = gamma < 1.0 ? pn.gradientl2Norm() * (1 - gamma) * (1 - gamma) : 0.0;
				l2norm += gamma > 0.0 ? dt.Squaredl2NormUnsafe() * gamma * gamma : 0.0;
				double alpha = loss / l2norm;
		        //System.out.println(alpha + " " + l2norm);
		        //Utils.block();
				
				if (gamma > 0.0) {
					//System.out.println(alpha + " " + C);
					//Utils.block();
					double lr = Math.min(alpha, C);
					for (int i = 0, L = dt.size(); i < L; ++i) {
						int x = dt.x(i);
						double g = dt.value(i) * gamma;
						params[x] += lr * g;
						total[x] += lr * updCnt * g;
					}
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
	
	public void averageParameters() {
		back = params;
		double[] avgParams = new double[size];
		for (int i = 0; i < size; ++i) {
			avgParams[i] = (params[i] * (updCnt+1) - total[i])/updCnt;			
		}		
		params = avgParams;

		pn.averageParameters(updCnt);
	}
	
	public void unaverageParameters() {
		params = back;
		pn.unaverageParameters();
	}

	public void clearTheta() 
	{
		params = new double[size];
		total = new double[size];
	}
	
	public void printNorm() {
		double norm = Utils.squaredSum(params);
		System.out.println("squared norm: " + norm);
	}
}
