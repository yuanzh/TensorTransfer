package parser.tensor;

import java.util.Arrays;

import parser.DependencyInstance;
import parser.Options;
import parser.TensorTransfer;
import utils.FeatureVector;
import utils.Utils;

public class HierarchicalFeatureNode extends FeatureNode {
	public FeatureDataItem[] headLexicalData;
	public FeatureDataItem[] modLexicalData;
	public FeatureDataItem[] headContextData;
	public FeatureDataItem[] modContextData;
	public FeatureDataItem[] labelData;
	public FeatureDataItem[] headData;
	public FeatureDataItem[] modData;
	public FeatureDataItem[] ddData;
	
	public FeatureDataItem emptyLabelData;
	
	double[] typoScore;
	double[] arcScore;
	double[] finalScore;
	double[] gScore;
	double[] g2Score;
	double[] g3Score;
	
	public static final double svoWeight = 1.0 / 3;
	public static final double typoWeight = 1.0 / 3;
	public static final double multiWeight = 1.0 / 3;
	
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
			delexical = pn.node[1];
			
			headLexicalData = new FeatureDataItem[n];
			modLexicalData = new FeatureDataItem[n];
			for (int i = 0; i < n; ++i) {
				ParameterNode hpn = pn.node[0].node[0];
				ParameterNode mpn = pn.node[0].node[1];
				int lexRank = hpn.rank;
				Utils.Assert(mpn.rank == lexRank);
				FeatureVector fv = pipe.ff.createLexicalFeatures(inst, i, hpn.featureSize, hpn.featureBias);
				double[] headScore = new double[lexRank];
				double[] modScore = new double[lexRank];
				
				for (int r = 0; r < lexRank; ++r) {
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
			
//			for (int z = 0; z < fv.size(); ++z) {
//				Utils.Assert(hpn.isActive[fv.x(z)]);
//				if (i > 0)
//					Utils.Assert(mpn.isActive[fv.x(z)]);
//			}
			
			headContextData[i] = new FeatureDataItem(fv, headScore);
			modContextData[i] = new FeatureDataItem(fv, modScore);
			
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
			
			headData[i] = new FeatureDataItem(fv, headScore);
			modData[i] = new FeatureDataItem(fv, modScore);
		}
		
		// direction and distance and typo
		int d = ParameterNode.d;
		ddData = new FeatureDataItem[2 * d];
		ParameterNode dpn = options.learnLabel ? delexical.node[2].node[1].node[2] 
				: delexical.node[2].node[2];
		Utils.Assert(rank == dpn.rank);
		for (int i = 0; i < 2 * d; ++i) {
			FeatureVector fv = pipe.fr.getDDTypoFv(i, inst.lang);
			double[] score = new double[rank];
			for (int r = 0; r < rank; ++r) {
				score[r] = fv.dotProduct(dpn.param[r]);
			}
//			for (int z = 0; z < fv.size(); ++z) {
//				Utils.Assert(dpn.isActive[fv.x(z)]);
//			}
			ddData[i] = new FeatureDataItem(fv, score);
		}
		
		// label
		if (options.learnLabel) {
			ParameterNode lpn = delexical.node[2].node[0];
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
		
		double[] lexScore = null;
		if (options.lexical) {
			int rank = pn.rank;
			int rank2 = options.extraR;
			lexScore = new double[rank];
			
			double[] headLexicalScore = headLexicalData[h].score;
			double[] modLexicalScore = modLexicalData[m].score;
			for (int r = 0; r < rank; ++r) {
				int st = r * rank2;
				for (int r2 = 0; r2 < rank2; ++r2) {
					lexScore[r] += headLexicalScore[st + r2] * modLexicalScore[st + r2];
				}
			}
		}

		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		//double[] tScore = Utils.dot(hScore, mScore, ddScore);
		double[] tScore = Utils.dot_s(typoScore, hScore, mScore, ddScore);
		
		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		FeatureVector fv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		//Utils.Assert(fv.size() == 0);
		ParameterNode tpn = options.learnLabel ? delexical.node[2].node[1] : delexical.node[2];
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += fv.dotProduct(tpn.param[r]);
		}
		
		if (options.learnLabel) {
			double[] lScore = l < 0 ? emptyLabelData.score : labelData[l].score;
			//double[] arcScore = Utils.dot(lScore, tScore);
			double[] aScore = Utils.dot_s(arcScore, lScore, tScore);
			fv = pipe.fr.getSVOFv(hp, mp, l, binDist, lang);
			//Utils.Assert(fv.size() == 0);
			ParameterNode apn = delexical.node[2];
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] += fv.dotProduct(apn.param[r]);
			tScore = aScore;
		}
		
		//return Utils.sum(Utils.dot(hcScore, mcScore, tScore));
		if (options.lexical) 
			return Utils.sum(Utils.dot_s(finalScore, lexScore, hcScore, mcScore, tScore));
		else
			return Utils.sum(Utils.dot_s(finalScore, hcScore, mcScore, tScore));
	}
	/*
	@Override
	public double getScore(int h, int m, int label) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		double[] lScore = options.learnLabel ? labelData[label].score : null;

		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		FeatureVector tfv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		ParameterNode tpn = options.learnLabel ? delexical.node[2].node[1] : delexical.node[2];
		double[] tScore = new double[tpn.rank];
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] = tfv.dotProduct(tpn.param[r]);
		}
		
		double[] aScore = null;
		if (options.learnLabel) {
			FeatureVector afv = pipe.fr.getSVOFv(hp, mp, label, binDist, lang);
			ParameterNode apn = delexical.node[2];
			aScore = new double[apn.rank];
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] = afv.dotProduct(apn.param[r]);
		}
		
		double sum = 0.0;
		for (int r = 0, rank = pn.rank; r < rank; ++r) {
			sum += hcScore[r] * mcScore[r] * hScore[r] * mScore[r] * ddScore[r] * (options.learnLabel ? lScore[r] : 1.0) * multiWeight
					+ hcScore[r] * mcScore[r] * tScore[r] * (options.learnLabel ? lScore[r] : 1.0) * typoWeight
					+ (options.learnLabel ? hcScore[r] * mcScore[r] * aScore[r] : 0.0) * svoWeight;
		}
		
		return sum;
	}
	*/
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
/*
	@Override
	public double addGradient(int h, int m, int label, double val, ParameterNode pn) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		double[] lScore = options.learnLabel ? labelData[label].score : null;

		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		ParameterNode hcpn = delexical.node[0];
		ParameterNode mcpn = delexical.node[1];
		ParameterNode apn = options.learnLabel ? delexical.node[2] : null;
		ParameterNode lpn = options.learnLabel ? apn.node[0] : null;
		ParameterNode tpn = options.learnLabel ? apn.node[1] : delexical.node[2];
		ParameterNode hpn = tpn.node[0];
		ParameterNode mpn = tpn.node[1];
		ParameterNode dpn = tpn.node[2];
		
		FeatureVector tfv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		FeatureVector afv = options.learnLabel ? pipe.fr.getSVOFv(hp, mp, label, binDist, lang) : null;
		
		double[] tScore = new double[tpn.rank];
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] = tfv.dotProduct(tpn.param[r]);
		}
		
		double[] aScore = null;
		if (options.learnLabel) {
			aScore = new double[apn.rank];
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] = afv.dotProduct(apn.param[r]);
		}
		
		double score = 0.0;
		for (int r = 0, rank = pn.rank; r < rank; ++r) {
			score += hcScore[r] * mcScore[r] * hScore[r] * mScore[r] * ddScore[r] * (options.learnLabel ? lScore[r] : 1.0) * multiWeight
					+ hcScore[r] * mcScore[r] * tScore[r] * (options.learnLabel ? lScore[r] : 1.0) * typoWeight
					+ (options.learnLabel ? hcScore[r] * mcScore[r] * aScore[r] : 0.0) * svoWeight;
		}
		
		// update h, m and dd
		for (int r = 0, rank = pn.rank; r < rank; ++r) {
			double g = val * hcScore[r] * mcScore[r] * (options.learnLabel ? lScore[r] : 1.0) * multiWeight;
			hpn.dFV[r].addEntries(headData[h].fv, g * mScore[r] * ddScore[r]);
			mpn.dFV[r].addEntries(modData[m].fv, g * hScore[r] * ddScore[r]);
			dpn.dFV[r].addEntries(ddData[binDist].fv, g * hScore[r] * mScore[r]);
		}

		if (options.learnLabel) {
			// update l
			for (int r = 0, rank = pn.rank; r < rank; ++r) {
				double g = val * (hScore[r] * mScore[r] * ddScore[r] * multiWeight
							+ tScore[r] * typoWeight)
							* hcScore[r] * mcScore[r];
				lpn.dFV[r].addEntries(labelData[label].fv, g);
			}
			
			// update svo
			for (int r = 0, rank = pn.rank; r < rank; ++r) {
				double g = val * hcScore[r] * mcScore[r] * svoWeight;
				apn.dFV[r].addEntries(afv, g);
			}
		}
		
		// update typo
		for (int r = 0, rank = pn.rank; r < rank; ++r) {
			double g = val * hcScore[r] * mcScore[r] * (options.learnLabel ? lScore[r] : 1.0) * typoWeight;
			tpn.dFV[r].addEntries(tfv, g);
		}
		
		// update hc and mc
		for (int r = 0, rank = pn.rank; r < rank; ++r) {
			double g = val * ((hScore[r] * mScore[r] * ddScore[r] * multiWeight + tScore[r] * typoWeight)
					* (options.learnLabel ? lScore[r] : 1.0) + (options.learnLabel ? aScore[r] * svoWeight : 0.0));
			hcpn.dFV[r].addEntries(headContextData[h].fv, g * mcScore[r]);
			mcpn.dFV[r].addEntries(modContextData[m].fv, g * hcScore[r]);
		}
		
		return score;
	}
	*/

	@Override
	public double addGradient(int h, int m, int label, double val, ParameterNode pn) {
		int hp = inst.postagids[h];
		int mp = inst.postagids[m];
		int binDist = pipe.ff.getBinnedDistance(h - m);
		int lang = inst.lang;
		
		double[] v = new double[pn.rank];
		Arrays.fill(v, val);
		
		double[] g = null; 

		double[] hcScore = headContextData[h].score;
		double[] mcScore = modContextData[m].score;
		
		double[] hScore = headData[h].score;
		double[] mScore = modData[m].score;
		double[] ddScore = ddData[binDist].score;
		//double[] tScore = Utils.dot(hScore, mScore, ddScore);
		double[] tScore = Utils.dot_s(typoScore, hScore, mScore, ddScore);
		
		ParameterNode delexical = options.lexical ? pn.node[1] : pn;
		ParameterNode hcpn = delexical.node[0];
		ParameterNode mcpn = delexical.node[1];
		ParameterNode apn = options.learnLabel ? delexical.node[2] : null;
		ParameterNode lpn = options.learnLabel ? apn.node[0] : null;
		ParameterNode tpn = options.learnLabel ? apn.node[1] : delexical.node[2];
		ParameterNode hpn = tpn.node[0];
		ParameterNode mpn = tpn.node[1];
		ParameterNode dpn = tpn.node[2];

		double[] lexScore = null;
		ParameterNode hlpn = options.lexical ? pn.node[0].node[0] : null;
		ParameterNode mlpn = options.lexical ? pn.node[0].node[1] : null;
		double[] headLexicalScore = options.lexical ? headLexicalData[h].score : null;
		double[] modLexicalScore = options.lexical ? modLexicalData[m].score : null;
		
		if (options.lexical) {
			int rank = pn.rank;
			int rank2 = options.extraR;
			lexScore = new double[rank];
			
			for (int r = 0; r < rank; ++r) {
				int st = r * rank2;
				for (int r2 = 0; r2 < rank2; ++r2) {
					lexScore[r] += headLexicalScore[st + r2] * modLexicalScore[st + r2];
				}
			}
		}

		FeatureVector tfv = pipe.fr.getTypoFv(hp, mp, binDist, lang);
		//Utils.Assert(tfv.size() == 0);
		for (int r = 0; r < tpn.rank; ++r) {
			tScore[r] += tfv.dotProduct(tpn.param[r]);
		}
		
		if (options.learnLabel) {
			double[] lScore = label < 0 ? emptyLabelData.score : labelData[label].score;
			//double[] aScore = Utils.dot(lScore, tScore);
			double[] aScore = Utils.dot_s(arcScore, lScore, tScore);
			FeatureVector afv = pipe.fr.getSVOFv(hp, mp, label, binDist, lang);
			//Utils.Assert(afv.size() == 0);
			for (int r = 0; r < apn.rank; ++r)
				aScore[r] += afv.dotProduct(apn.param[r]);
			
			if (options.lexical) {
				int rank = pn.rank;
				int rank2 = options.extraR;

				for (int r = 0; r < rank; ++r) {
					double tmp = v[r] * hcScore[r] * mcScore[r] * aScore[r];
					int st = r * rank2;
					for (int r2 = 0; r2 < rank2; ++r2) {
						// update head lexical
						double dv = tmp * modLexicalScore[st + r2];
						hlpn.dFV[st + r2].addEntries(headLexicalData[h].fv, dv);
						
						// update mod lexical
						dv = tmp * headLexicalScore[st + r2];
						mlpn.dFV[st + r2].addEntries(modLexicalData[m].fv, dv);
					}
				}
			}

			// update head context
			//g = Utils.dot(v, mcScore, aScore);
			g = options.lexical ? Utils.dot_s(gScore, v, lexScore, mcScore, aScore)
					: Utils.dot_s(gScore, v, mcScore, aScore);
			for (int r = 0; r < hcpn.rank; ++r) {
				hcpn.dFV[r].addEntries(headContextData[h].fv, g[r]);
			}
			
			// update mod context
			//g = Utils.dot(v, hcScore, aScore);
			g = options.lexical ? Utils.dot_s(gScore, v, lexScore, hcScore, aScore)
					: Utils.dot_s(gScore, v, hcScore, aScore);
			for (int r = 0; r < mcpn.rank; ++r) {
				mcpn.dFV[r].addEntries(modContextData[m].fv, g[r]);
			}
			
			// update svo
			//g = Utils.dot(v, hcScore, mcScore);
			g = options.lexical ? Utils.dot_s(gScore, v, lexScore, hcScore, mcScore)
					: Utils.dot_s(gScore, v, hcScore, mcScore);
			for (int r = 0; r < apn.rank; ++r) {
				apn.dFV[r].addEntries(afv, g[r]);
			}
			
			// update label
			//double[] g2 = Utils.dot(g, tScore);
			double[] g2 = Utils.dot_s(g2Score, g, tScore);
			for (int r = 0; r < lpn.rank; ++r) {
				FeatureVector fv = label < 0 ? emptyLabelData.fv : labelData[label].fv;
				lpn.dFV[r].addEntries(fv, g2[r]);
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
				hpn.dFV[r].addEntries(headData[h].fv, g3[r]);
			}
			
			// update mod
			//g3 = Utils.dot(g2, hScore, ddScore);
			g3 = Utils.dot_s(g3Score, g2, hScore, ddScore);
			for (int r = 0; r < mpn.rank; ++r) {
				mpn.dFV[r].addEntries(modData[m].fv, g3[r]);
			}
			
			// update dd
			//g3 = Utils.dot(g2, hScore, mScore);
			g3 = Utils.dot_s(g3Score, g2, hScore, mScore);
			for (int r = 0; r < dpn.rank; ++r) {
				dpn.dFV[r].addEntries(ddData[binDist].fv, g3[r]);
			}
			
			return Utils.sum(Utils.dot_s(finalScore, hcScore, mcScore, aScore));
		}
		else {
			// update head context
			g = Utils.dot(v, mcScore, tScore);
			for (int r = 0; r < hcpn.rank; ++r) {
				hcpn.dFV[r].addEntries(headContextData[h].fv, g[r]);
			}
			
			// update mod context
			g = Utils.dot(v, hcScore, tScore);
			for (int r = 0; r < mcpn.rank; ++r) {
				mcpn.dFV[r].addEntries(modContextData[m].fv, g[r]);
			}
			
			// update typo
			g = Utils.dot(v, hcScore, mcScore);
			for (int r = 0; r < tpn.rank; ++r) {
				tpn.dFV[r].addEntries(tfv, g[r]);
			}
			
			// update head
			double[] g2 = Utils.dot(g, mScore, ddScore);
			for (int r = 0; r < hpn.rank; ++r) {
				hpn.dFV[r].addEntries(headData[h].fv, g2[r]);
			}
			
			// update mod
			g2 = Utils.dot(g, hScore, ddScore);
			for (int r = 0; r < mpn.rank; ++r) {
				mpn.dFV[r].addEntries(modData[m].fv, g2[r]);
			}
			
			// update dd
			g2 = Utils.dot(g, hScore, mScore);
			for (int r = 0; r < dpn.rank; ++r) {
				dpn.dFV[r].addEntries(ddData[binDist].fv, g2[r]);
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
