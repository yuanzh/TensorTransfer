package parser.tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import parser.Options;
import parser.Options.UpdateMode;
import utils.Utils;

public class LowRankParam {
	ParameterNode pn;
	Options options;
	public ArrayList<MatrixEntry> list;
	
	public LowRankParam(ParameterNode pn) 
	{
		this.pn = pn;
		options = pn.options;
		list = new ArrayList<MatrixEntry>();
	}
	
	public void putEntry(int h, int m, int dd, int label, double v)
	{
		list.add(new MatrixEntry(h, m, dd, label, v));
	}
	
	public void putEntry(int h, int m, int hc, int mc, int dd, int label, double v)
	{
		list.add(new MultiwayEntry(h, m, hc, mc, dd, label, v));
	}
	
	public void putEntry(int h, int m, int hc, int mc, int dd, int label, int svo, int t, int hl, int ml, double v)
	{
		list.add(new HierarchicalEntry(h, m, hc, mc, dd, label, svo, t, hl, ml, v));
	}
	
	public void decomposeThreeway()
	{
		int maxRank = pn.rank;
    	ParameterNode hpn = pn.node[0];
    	ParameterNode mpn = pn.node[1];
    	ParameterNode dpn = pn.node[2];
    	ParameterNode lpn = options.learnLabel ? pn.node[3] : null;
		
		int MAXITER=1000;
		double eps = 1e-6;
		Random rnd = new Random(0);
		for (int i = 0; i < maxRank; ++i) {
			double[] h = new double[hpn.featureSize], m = new double[mpn.featureSize], dd = new double[dpn.featureSize];
			double[] l = options.learnLabel ? new double[lpn.featureSize] : null;
			for (int j = 0; j < mpn.featureSize; ++j)
				if (mpn.isActive[j])
					m[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < dpn.featureSize; ++j)
				if (dpn.isActive[j])
					dd[j] = rnd.nextDouble() - 0.5;
			Utils.normalize(m);
			Utils.normalize(dd);
			if (options.learnLabel) {
				for (int j = 0; j < lpn.featureSize; ++j)
					if (lpn.isActive[j])
						l[j] = rnd.nextDouble() - 0.5;
				Utils.normalize(l);
			}
			
			int iter = 0;
			double norm = 0.0, lastnorm = Double.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				
				for (int j = 0; j < hpn.featureSize; ++j)
					h[j] = 0;
				for (MatrixEntry e : list) {
					h[e.h] += e.value * m[e.m] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < hpn.featureSize; ++k)
						h[k] -= dot * hpn.param[j][k];
				}
				Utils.normalize(h);
				
				for (int j = 0; j < mpn.featureSize; ++j)
					m[j] = 0;
				for (MatrixEntry e : list) {
					m[e.m] += e.value * h[e.h] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < mpn.featureSize; ++k)
						m[k] -= dot * mpn.param[j][k];
				}
				Utils.normalize(m);
				
				for (int j = 0; j < dpn.featureSize; ++j)
					dd[j] = 0;
				for (MatrixEntry e : list) {
					dd[e.dd] += e.value * h[e.h] * m[e.m] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(m, mpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < dpn.featureSize; ++k)
						dd[k] -= dot * dpn.param[j][k];
				}
				
				if (options.learnLabel) {
					Utils.normalize(dd);
					
					for (int j = 0; j < lpn.featureSize; ++j)
						l[j] = 0;
					for (MatrixEntry e : list) {
						l[e.label] += e.value * h[e.h] * m[e.m] * dd[e.dd];
					}
					for (int j = 0; j < i; ++j) {
						double dot = Utils.dotsum(h, hpn.param[j]) 
								   * Utils.dotsum(m, mpn.param[j])
								   * Utils.dotsum(dd, dpn.param[j]);
						for (int k = 0; k < lpn.featureSize; ++k)
							l[k] -= dot * lpn.param[j][k];
					}

					norm = Math.sqrt(Utils.squaredSum(l));
					if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
						break;
					lastnorm = norm;
					
				}
				else {
					norm = Math.sqrt(Utils.squaredSum(dd));
					if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
						break;
					lastnorm = norm;
				}
				
			}
			if (iter >= MAXITER) {
				System.out.printf("\tWARNING: Power method didn't converge." +
						"R=%d sigma=%f%n", i, norm);
			}
			if (Math.abs(norm) <= eps) {
				System.out.printf("\tWARNING: Power method has nearly-zero sigma. R=%d%n",i);
			}
			System.out.printf("\t%.2f", norm);
			hpn.param[i] = h;
			mpn.param[i] = m;
			dpn.param[i] = dd;
			if (options.learnLabel)
				lpn.param[i] = l;
		}
		
		if (options.updateMode == UpdateMode.MIRA) {
			for (int i = 0; i < maxRank; ++i) {
				hpn.total[i] = hpn.param[i].clone();
				mpn.total[i] = mpn.param[i].clone();
				dpn.total[i] = dpn.param[i].clone();
				if (options.learnLabel)
					lpn.total[i] = lpn.param[i].clone();
			}
		}
	}
	
	public void decomposeMultiway()
	{
		int maxRank = pn.rank;
		ParameterNode hpn = pn.node[0];
		ParameterNode mpn = pn.node[1];
		ParameterNode hcpn = pn.node[2];
		ParameterNode mcpn = pn.node[3];
		ParameterNode dpn = pn.node[4];
		ParameterNode lpn = options.learnLabel ? pn.node[5] : null;
		
		int MAXITER=1000;
		double eps = 1e-6;
		Random rnd = new Random(0);
		for (int i = 0; i < maxRank; ++i) {
			double[] h = new double[hpn.featureSize], m = new double[mpn.featureSize], dd = new double[dpn.featureSize];
			double[] hc = new double[hcpn.featureSize], mc = new double[mcpn.featureSize];
			double[] l = options.learnLabel ? new double[lpn.featureSize] : null;
			for (int j = 0; j < mpn.featureSize; ++j)
				if (mpn.isActive[j])
					m[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < hcpn.featureSize; ++j)
				if (hcpn.isActive[j])
					hc[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < mcpn.featureSize; ++j)
				if (mcpn.isActive[j])
					mc[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < dpn.featureSize; ++j)
				if (dpn.isActive[j])
					dd[j] = rnd.nextDouble() - 0.5;
			Utils.normalize(m);
			Utils.normalize(hc);
			Utils.normalize(mc);
			Utils.normalize(dd);
			if (options.learnLabel) {
				for (int j = 0; j < lpn.featureSize; ++j)
					if (lpn.isActive[j])
						l[j] = rnd.nextDouble() - 0.5;
				Utils.normalize(l);
			}
			
			int iter = 0;
			double norm = 0.0, lastnorm = Double.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				
				for (int j = 0; j < hpn.featureSize; ++j)
					h[j] = 0;
				for (MatrixEntry ee : list) {
					MultiwayEntry e = (MultiwayEntry)ee;
					h[e.h] += e.value * m[e.m] * hc[e.hc] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < hpn.featureSize; ++k)
						h[k] -= dot * hpn.param[j][k];
				}
				Utils.normalize(h);
				
				for (int j = 0; j < mpn.featureSize; ++j)
					m[j] = 0;
				for (MatrixEntry ee : list) {
					MultiwayEntry e = (MultiwayEntry)ee;
					m[e.m] += e.value * h[e.h] * hc[e.hc] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < mpn.featureSize; ++k)
						m[k] -= dot * mpn.param[j][k];
				}
				Utils.normalize(m);
				
				for (int j = 0; j < hcpn.featureSize; ++j)
					hc[j] = 0;
				for (MatrixEntry ee : list) {
					MultiwayEntry e = (MultiwayEntry)ee;
					hc[e.hc] += e.value * m[e.m] * h[e.h] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(h, hpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < hcpn.featureSize; ++k)
						hc[k] -= dot * hcpn.param[j][k];
				}
				Utils.normalize(hc);
				
				for (int j = 0; j < mcpn.featureSize; ++j)
					mc[j] = 0;
				for (MatrixEntry ee : list) {
					MultiwayEntry e = (MultiwayEntry)ee;
					mc[e.mc] += e.value * h[e.h] * hc[e.hc] * m[e.m] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(m, mpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < mcpn.featureSize; ++k)
						mc[k] -= dot * mcpn.param[j][k];
				}
				Utils.normalize(mc);
				
				for (int j = 0; j < dpn.featureSize; ++j)
					dd[j] = 0;
				for (MatrixEntry ee : list) {
					MultiwayEntry e = (MultiwayEntry)ee;
					dd[e.dd] += e.value * h[e.h] * m[e.m] * hc[e.hc] * mc[e.mc] * (e.label >= 0 ? l[e.label] : 1.0);
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(m, mpn.param[j])
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < dpn.featureSize; ++k)
						dd[k] -= dot * dpn.param[j][k];
				}
				
				if (options.learnLabel) {
					Utils.normalize(dd);
					
					for (int j = 0; j < lpn.featureSize; ++j)
						l[j] = 0;
					for (MatrixEntry ee : list) {
						MultiwayEntry e = (MultiwayEntry)ee;
						l[e.label] += e.value * h[e.h] * m[e.m] * hc[e.hc] * mc[e.mc] * dd[e.dd];
					}
					for (int j = 0; j < i; ++j) {
						double dot = Utils.dotsum(h, hpn.param[j]) 
								   * Utils.dotsum(m, mpn.param[j])
								   * Utils.dotsum(hc, hcpn.param[j])
								   * Utils.dotsum(mc, mcpn.param[j])
								   * Utils.dotsum(dd, dpn.param[j]);
						for (int k = 0; k < lpn.featureSize; ++k)
							l[k] -= dot * lpn.param[j][k];
					}

					norm = Math.sqrt(Utils.squaredSum(l));
					if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
						break;
					lastnorm = norm;
					
				}
				else {
					norm = Math.sqrt(Utils.squaredSum(dd));
					if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
						break;
					lastnorm = norm;
				}
				
			}
			if (iter >= MAXITER) {
				System.out.printf("\tWARNING: Power method didn't converge." +
						"R=%d sigma=%f%n", i, norm);
			}
			if (Math.abs(norm) <= eps) {
				System.out.printf("\tWARNING: Power method has nearly-zero sigma. R=%d%n",i);
			}
			System.out.printf("\t%.2f", norm);
			hpn.param[i] = h;
			mpn.param[i] = m;
			hcpn.param[i] = hc;
			mcpn.param[i] = mc;
			dpn.param[i] = dd;
			if (options.learnLabel)
				lpn.param[i] = l;
		}
		
		if (options.updateMode == UpdateMode.MIRA) {
			for (int i = 0; i < maxRank; ++i) {
				hpn.total[i] = hpn.param[i].clone();
				mpn.total[i] = mpn.param[i].clone();
				hcpn.total[i] = hcpn.param[i].clone();
				mcpn.total[i] = mcpn.param[i].clone();
				dpn.total[i] = dpn.param[i].clone();
				if (options.learnLabel)
					lpn.total[i] = lpn.param[i].clone();
			}
		}
	}

	public void decomposeHierarchicalway()
	{
		int maxRank = pn.rank;
		int rank2 = options.extraR;
		
    	ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		ParameterNode hcpn = delexical.node[0];
		ParameterNode mcpn = delexical.node[1];
		ParameterNode apn = options.learnLabel ? delexical.node[2] : null;
		ParameterNode lpn = options.learnLabel ? apn.node[0] : null;
		ParameterNode tpn = options.learnLabel ? apn.node[1] : delexical.node[2];
		ParameterNode hpn = tpn.node[0];
		ParameterNode mpn = tpn.node[1];
		ParameterNode dpn = tpn.node[2];
		ParameterNode hlpn = options.lexical ? pn.node[0].node[0] : null;
		ParameterNode mlpn = options.lexical ? pn.node[0].node[1] : null;
		
		int MAXITER=1000;
		double eps = 1e-6;
		Random rnd = new Random(0);
		for (int i = 0; i < maxRank; ++i) {
			double[] h = new double[hpn.featureSize], m = new double[mpn.featureSize], dd = new double[dpn.featureSize];
			double[] hc = new double[hcpn.featureSize], mc = new double[mcpn.featureSize];
			double[] l = options.learnLabel ? new double[lpn.featureSize] : null;
			double[] svo = options.learnLabel ? new double[apn.featureSize] : null;
			double[] t = new double[tpn.featureSize];
			double[][] hl = options.lexical ? new double[rank2][hlpn.featureSize] : null;
			double[][] ml = options.lexical ? new double[rank2][mlpn.featureSize] : null;
			
//			int split = 2;
			double norma = options.learnLabel ? 1.0 / 3 : 1.0;
			double norml = options.learnLabel ? Math.sqrt(1.0 - norma) : 1.0;
			double normt = options.learnLabel ? norml / 2 : 1.0 / 2;
			double normh = Math.cbrt(norml - normt);
			double normLex = Math.sqrt(1.0 / rank2);
			
			//double norml = 1.0;
			//double normh = 1.0;
			
			for (int j = 0; j < hpn.featureSize; ++j)
				if (hpn.isActive[j])
					h[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < mpn.featureSize; ++j)
				if (mpn.isActive[j])
					m[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < hcpn.featureSize; ++j)
				if (hcpn.isActive[j])
					hc[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < mcpn.featureSize; ++j)
				if (mcpn.isActive[j])
					mc[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < dpn.featureSize; ++j)
				if (dpn.isActive[j])
					dd[j] = rnd.nextDouble() - 0.5;
			for (int j = 0; j < tpn.featureSize; ++j)
				if (tpn.isActive[j])
					t[j] = rnd.nextDouble() - 0.5;
			Utils.normalize(h, normh);
			Utils.normalize(m, normh);
			Utils.normalize(hc);
			Utils.normalize(mc);
			Utils.normalize(dd, normh);
			Utils.normalize(t, normt);
			
			if (options.learnLabel) {
				for (int j = 0; j < apn.featureSize; ++j)
					if (apn.isActive[j])
						svo[j] = rnd.nextDouble() - 0.5;
				for (int j = 0; j < lpn.featureSize; ++j)
					if (lpn.isActive[j])
						l[j] = rnd.nextDouble() - 0.5;
				Utils.normalize(svo, norma);
				Utils.normalize(l, norml);
			}
			
			if (options.lexical) {
				for (int j = 0; j < hlpn.featureSize; ++j)
					if (hlpn.isActive[j]) 
						for (int k = 0; k < rank2; ++k)
							hl[k][j] = rnd.nextDouble() - 0.5;
				for (int j = 0; j < mlpn.featureSize; ++j)
					if (mlpn.isActive[j]) 
						for (int k = 0; k < rank2; ++k)
							ml[k][j] = rnd.nextDouble() - 0.5;
				for (int k = 0; k < rank2; ++k) {
					Utils.normalize(hl[k], normLex);
					Utils.normalize(ml[k], normLex);
				}
			}
			
			int iter = 0;
			double svoNorm = 0.0, typoNorm = 0.0, multiNorm = 0.0; 
			double lastSVONorm = Double.POSITIVE_INFINITY, lastTypoNorm = Double.POSITIVE_INFINITY, lastMultiNorm = Double.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				
				double[] lexSum = new double[i];
				Arrays.fill(lexSum, 1.0);
				
				if (options.lexical) {
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						if (e.h >= 0) {
							Utils.Assert(e.t < 0);
							Utils.Assert(e.svo < 0);
							e.delexScore = e.value * m[e.m] * h[e.h] * hc[e.hc] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
						}
						else if (e.t >= 0) {
							Utils.Assert(e.svo < 0);
							e.delexScore = e.value * hc[e.hc] * mc[e.mc] * t[e.t] * (e.label >= 0 ? l[e.label] : 1.0);
						}
						else if (e.svo >= 0) {
							Utils.Assert(e.t < 0);
							Utils.Assert(e.label < 0);
							e.delexScore = e.value * hc[e.hc] * mc[e.mc] * svo[e.svo];
						}
					}
					
					double[] delexSum = new double[i];
					for (int j = 0; j < i; ++j)
						delexSum[j] = ((Utils.dotsum(m, mpn.param[j]) 
								   * Utils.dotsum(h, hpn.param[j])
								   * Utils.dotsum(dd, dpn.param[j])
								   + Utils.dotsum(t, tpn.param[j])
									)
								   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0)
								   + (apn != null ? Utils.dotsum(svo, apn.param[j]) : 0.0)
								   )
								   * Utils.dotsum(mc, mcpn.param[j])
								   * Utils.dotsum(hc, hcpn.param[j]);
					
					// update head lexical
					for (int j = 0; j < hlpn.featureSize; ++j)
						for (int k = 0; k < rank2; ++k)
							hl[k][j] = 0;
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						for (int k = 0; k < rank2; ++k)
							hl[k][e.hl] += ml[k][e.ml] * e.delexScore;
					}
					for (int j = 0; j < i; ++j) {
						int st = j * rank2;
						for (int r2 = 0; r2 < rank2; ++r2) {
							double dot = delexSum[j] * Utils.dotsum(ml[r2], mlpn.param[st + r2]);
							for (int k = 0; k < hlpn.featureSize; ++k)
								hl[r2][k] -= dot * hlpn.param[st + r2][k];
						}
					}
					for (int r2 = 0; r2 < rank2; ++r2)
						Utils.normalize(hl[r2], normLex);

					// update mod lexical
					for (int j = 0; j < mlpn.featureSize; ++j)
						for (int k = 0; k < rank2; ++k)
							ml[k][j] = 0;
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						for (int k = 0; k < rank2; ++k)
							ml[k][e.ml] += hl[k][e.hl] * e.delexScore;
					}
					for (int j = 0; j < i; ++j) {
						int st = j * rank2;
						for (int r2 = 0; r2 < rank2; ++r2) {
							double dot = delexSum[j] * Utils.dotsum(hl[r2], hlpn.param[st + r2]);
							for (int k = 0; k < mlpn.featureSize; ++k)
								ml[r2][k] -= dot * mlpn.param[st + r2][k];
						}
					}
					for (int r2 = 0; r2 < rank2; ++r2)
						Utils.normalize(ml[r2], normLex);
					
					// compute lexical score
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						double sum = 0.0;
						for (int k = 0; k < rank2; ++k)
							sum += hl[k][e.hl] * ml[k][e.ml];
						e.lexScore = sum;
					}
					
					for (int j = 0; j < i; ++j) {
						int st = j * rank2;
						double sum = 0.0;
						for (int k = 0; k < rank2; ++k)
							sum += Utils.dotsum(hl[k], hlpn.param[st + k])
								* Utils.dotsum(ml[k], mlpn.param[st + k]);
						lexSum[j] = sum;
					}
				}
				
				
				for (int j = 0; j < hpn.featureSize; ++j)
					h[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.h >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.svo < 0);
						h[e.h] += e.value * e.lexScore * m[e.m] * hc[e.hc] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
					}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j]
							   * Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < hpn.featureSize; ++k)
						h[k] -= dot * hpn.param[j][k];
				}
				Utils.normalize(h, normh);
				
				for (int j = 0; j < mpn.featureSize; ++j)
					m[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.m >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.svo < 0);
						m[e.m] += e.value * e.lexScore * h[e.h] * hc[e.hc] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
					}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j]
							   * Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < mpn.featureSize; ++k)
						m[k] -= dot * mpn.param[j][k];
				}
				Utils.normalize(m, normh);
				
				for (int j = 0; j < dpn.featureSize; ++j)
					dd[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.dd >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.svo < 0);
						dd[e.dd] += e.value * e.lexScore * h[e.h] * m[e.m] * hc[e.hc] * mc[e.mc] * (e.label >= 0 ? l[e.label] : 1.0);
					}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j]
							   * Utils.dotsum(h, hpn.param[j]) 
							   * Utils.dotsum(m, mpn.param[j])
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < dpn.featureSize; ++k)
						dd[k] -= dot * dpn.param[j][k];
				}
				Utils.normalize(dd, normh);
				
				for (int j = 0; j < tpn.featureSize; ++j)
					t[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.t >= 0) {
						Utils.Assert(e.svo < 0);
						Utils.Assert(e.h < 0);
						t[e.t] += e.value * e.lexScore * hc[e.hc] * mc[e.mc] * (e.label >= 0 ? l[e.label] : 1.0);
					}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j]
							   * Utils.dotsum(hc, hcpn.param[j])
							   * Utils.dotsum(mc, mcpn.param[j])
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0);
					for (int k = 0; k < tpn.featureSize; ++k)
						t[k] -= dot * tpn.param[j][k];
				}
				Utils.normalize(t, normt);
				
				if (options.learnLabel) {
					
					for (int j = 0; j < apn.featureSize; ++j)
						svo[j] = 0;
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						if (e.svo >= 0) {
							//System.out.println("bbb: " + e.value + " " + hc[e.hc] + " " + mc[e.mc]);
							Utils.Assert(e.h < 0);
							Utils.Assert(e.t < 0);
							Utils.Assert(e.label < 0);
							svo[e.svo] += e.value * e.lexScore * hc[e.hc] * mc[e.mc];
						}
					}
					for (int j = 0; j < i; ++j) {
						double dot = lexSum[j] 
								   * Utils.dotsum(hc, hcpn.param[j])
								   * Utils.dotsum(mc, mcpn.param[j]);
						for (int k = 0; k < apn.featureSize; ++k)
							svo[k] -= dot * apn.param[j][k];
					}
					Utils.normalize(svo, norma);

					for (int j = 0; j < lpn.featureSize; ++j)
						l[j] = 0;
					for (MatrixEntry ee : list) {
						HierarchicalEntry e = (HierarchicalEntry)ee;
						if (e.label >= 0) {
							Utils.Assert(e.svo < 0);
							if (e.h >= 0) {
								Utils.Assert(e.t < 0);
								l[e.label] += e.value * e.lexScore * h[e.h] * m[e.m] * hc[e.hc] * mc[e.mc] * dd[e.dd];
							}
							else if (e.t >= 0) {
								l[e.label] += e.value * e.lexScore * hc[e.hc] * mc[e.mc] * t[e.t];
							}
							else {
								System.out.println("Warning 3: " + e.h + " " + e.m + " " + e.dd + " " + e.hc + " " + e.mc + " " + e.t + " " + e.svo + " " + e.label);
								Utils.ThrowException("aaa");
							}
						}
					}
					for (int j = 0; j < i; ++j) {
						double dot = lexSum[j]
								   * (Utils.dotsum(h, hpn.param[j]) 
								   * Utils.dotsum(m, mpn.param[j])
								   * Utils.dotsum(dd, dpn.param[j])
								   + Utils.dotsum(t, tpn.param[j])
								   )
								   * Utils.dotsum(hc, hcpn.param[j])
								   * Utils.dotsum(mc, mcpn.param[j]);
						for (int k = 0; k < lpn.featureSize; ++k)
							l[k] -= dot * lpn.param[j][k];
					}
					Utils.normalize(l, norml);
				}
				else {
				}
				
				for (int j = 0; j < hcpn.featureSize; ++j)
					hc[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.h >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.svo < 0);
						hc[e.hc] += e.value * e.lexScore * m[e.m] * h[e.h] * mc[e.mc] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
					}
					else if (e.t >= 0) {
						Utils.Assert(e.svo < 0);
						hc[e.hc] += e.value * e.lexScore * mc[e.mc] * t[e.t] * (e.label >= 0 ? l[e.label] : 1.0);
					}
					else if (e.svo >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.label < 0);
						hc[e.hc] += e.value * e.lexScore * mc[e.mc] * svo[e.svo];
					}
					else {
						System.out.println("Warning 1");
					}
					//if (e.hc == 0) {
					//	System.out.println("aaa: " + hc[e.hc] + " " +  e.h + " " + e.t + " " + e.svo + " " + mc[e.mc] + " " + t[e.t] + " " + l[e.label]);
					//}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j] 
							   * ((Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(h, hpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   + Utils.dotsum(t, tpn.param[j])
								)
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0)
							   + (apn != null ? Utils.dotsum(svo, apn.param[j]) : 0.0)
							   )
							   * Utils.dotsum(mc, mcpn.param[j]);
					for (int k = 0; k < hcpn.featureSize; ++k)
						hc[k] -= dot * hcpn.param[j][k];
				}
				Utils.normalize(hc);
				
				for (int j = 0; j < mcpn.featureSize; ++j)
					mc[j] = 0;
				for (MatrixEntry ee : list) {
					HierarchicalEntry e = (HierarchicalEntry)ee;
					if (e.h >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.svo < 0);
						mc[e.mc] += e.value * e.lexScore * h[e.h] * hc[e.hc] * m[e.m] * dd[e.dd] * (e.label >= 0 ? l[e.label] : 1.0);
					}
					else if (e.t >= 0) {
						Utils.Assert(e.svo < 0);
						mc[e.mc] += e.value * e.lexScore * hc[e.hc] * t[e.t] * (e.label >= 0 ? l[e.label] : 1.0);
					}
					else if (e.svo >= 0) {
						Utils.Assert(e.t < 0);
						Utils.Assert(e.label < 0);
						mc[e.mc] += e.value * e.lexScore * hc[e.hc] * svo[e.svo];
					}
					else {
						System.out.println("Warning 2");
					}
				}
				for (int j = 0; j < i; ++j) {
					double dot = lexSum[j]
							   * ((Utils.dotsum(m, mpn.param[j]) 
							   * Utils.dotsum(h, hpn.param[j])
							   * Utils.dotsum(dd, dpn.param[j])
							   + Utils.dotsum(t, tpn.param[j])
								)
							   * (lpn != null ? Utils.dotsum(l, lpn.param[j]) : 1.0)
							   + (apn != null ? Utils.dotsum(svo, apn.param[j]) : 0.0)
							   )
							   * Utils.dotsum(hc, hcpn.param[j]);
					for (int k = 0; k < mcpn.featureSize; ++k)
						mc[k] -= dot * mcpn.param[j][k];
				}
				//Utils.normalize(mc);

				svoNorm = Math.sqrt(Utils.squaredSum(svo));
				typoNorm = Math.sqrt(Utils.squaredSum(t));
				multiNorm = Math.sqrt(Utils.squaredSum(mc));
				if (lastSVONorm != Double.POSITIVE_INFINITY && Math.abs(svoNorm-lastSVONorm) < eps
					&& lastTypoNorm != Double.POSITIVE_INFINITY && Math.abs(typoNorm-lastTypoNorm) < eps
					&& lastMultiNorm != Double.POSITIVE_INFINITY && Math.abs(multiNorm-lastMultiNorm) < eps) {
					break;
				}
				lastSVONorm = svoNorm;
				lastTypoNorm = typoNorm;
				lastMultiNorm = multiNorm;
			}
			if (iter >= MAXITER) {
				System.out.printf("\tWARNING: Power method didn't converge." +
						"R=%d sigma=%f %f %f%n", i, svoNorm, typoNorm, multiNorm);
			}
			//if (Math.abs(svoNorm) <= eps || Math.abs(typoNorm) <= eps || Math.abs(multiNorm) <= eps) {
			//	System.out.printf("\tWARNING: Power method has nearly-zero sigma. R=%d%n",i);
			//}
			System.out.printf("\t%.2f/%.2f/%.2f", svoNorm, typoNorm, multiNorm);
			
			double scale = Math.sqrt(1.0 / multiNorm);
			for (int j = 0; j < hcpn.featureSize; ++j) {
				hc[j] /= scale;
				mc[j] *= scale;
			}
			double hn = Utils.squaredSum(hc);
			double mn = Utils.squaredSum(mc);
			Utils.Assert(Math.abs(hn -mn) < eps);
			
			hpn.param[i] = h;
			mpn.param[i] = m;
			hcpn.param[i] = hc;
			mcpn.param[i] = mc;
			dpn.param[i] = dd;
			tpn.param[i] = t;
			if (options.learnLabel) {
				lpn.param[i] = l;
				apn.param[i] = svo;
			}
			if (options.lexical) {
				int st = i * rank2;
				for (int r2 = 0; r2 < rank2; ++r2) {
					hlpn.param[st + r2] = hl[r2];
					mlpn.param[st + r2] = ml[r2];
				}
			}
		}
		
		if (options.updateMode == UpdateMode.MIRA) {
			for (int i = 0; i < maxRank; ++i) {
				hpn.total[i] = hpn.param[i].clone();
				mpn.total[i] = mpn.param[i].clone();
				hcpn.total[i] = hcpn.param[i].clone();
				mcpn.total[i] = mcpn.param[i].clone();
				dpn.total[i] = dpn.param[i].clone();
				tpn.total[i] = tpn.param[i].clone();
				if (options.learnLabel) {
					lpn.total[i] = lpn.param[i].clone();
					apn.total[i] = apn.param[i].clone();
				}
				if (options.lexical) {
					int st = i * rank2;
					for (int r2 = 0; r2 < rank2; ++r2) {
						hlpn.total[st + r2] = hlpn.param[st + r2].clone();
						mlpn.total[st + r2] = mlpn.param[st + r2].clone();
					}
				}
			}
		}
	}

}

class MatrixEntry
{
	public int h, m, dd, label;
	public double value;
	public MatrixEntry(int _h, int _m, int _dd, int _label, double _value)
	{
		h = _h;
		m = _m;
		dd = _dd;
		label = _label;
		value = _value;
	}
}

class MultiwayEntry extends MatrixEntry
{
	public int hc, mc;
	public MultiwayEntry(int _h, int _m, int _hc, int _mc, int _dd, int _label, double _value)
	{
		super(_h, _m, _dd, _label, _value);
		hc = _hc;
		mc = _mc;
	}
}

class HierarchicalEntry extends MatrixEntry
{
	public int hc, mc, svo, t, hl, ml;
	public double lexScore;
	public double delexScore;
	public HierarchicalEntry(int _h, int _m, int _hc, int _mc, int _dd, int _label, int _svo, int _t, int _hl, int _ml, double _value)
	{
		super(_h, _m, _dd, _label, _value);
		hc = _hc;
		mc = _mc;
		svo = _svo;
		t = _t;
		hl = _hl;
		ml = _ml;
		lexScore = 1.0;
		delexScore = 1.0;
	}
}
