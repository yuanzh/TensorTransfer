package parser.tensor;

import java.io.Serializable;

import parser.DependencyPipe;
import parser.Options;
import parser.Options.TensorMode;
import parser.Options.UpdateMode;
import utils.FeatureVector;
import utils.TypologicalInfo.TypoFeatureType;
import utils.Utils;
import utils.DictionarySet.DictionaryTypes;

public class ParameterNode implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public static final int d = 7;

	Options options;
	DependencyPipe pipe;
	
	public int rank;
	public double gamma;
	
	public int posNum;
	public int labelNum;

	// self feature
	public int featureSize;
	public int[] featureBias;		// [feature template num - 1]
	public double[][] param;		// [rank][feature Num]
	public transient boolean[] isActive;	// [feature num], whether the feature is used
	public transient FeatureVector[] dFV;	// [rank];
	
	// adaGrad
	private transient double[][] sg;	// [rank][feature num]
	private transient double adaAlpha;
	private transient double adaEps;
	
	// mira
	private transient double[][] total;
	private transient double[][] back;
	private transient double C;

	// hierarchical structure
	public int nodeNum;
	public ParameterNode[] node;
	
	public ParameterNode(Options options, DependencyPipe pipe, int rank) {
		this.options = options;
		this.pipe = pipe;
		this.rank = rank;
		gamma = options.gamma;

		posNum = pipe.dictionaries.size(DictionaryTypes.POS);
		labelNum = pipe.dictionaries.size(DictionaryTypes.DEPLABEL);
	}
	
	public void constructStructure() {
		if (options.tensorMode == TensorMode.Threeway)
			constructThreewayStructure();
		else if (options.tensorMode == TensorMode.Multiway)
			constructMultiwayStructure();
		else if (options.tensorMode == TensorMode.Hierarchical)
			constructHierarchicalStructure();
		else
			Utils.ThrowException("unsupported structure");
	}
	
	private void setNodeNum(int n) {
		nodeNum = n;
		if (n > 0) {
			node = new ParameterNode[nodeNum];
		}
	}
	
	private void setFeatureSizeAndBias(int[] dim) {
		int sum = 0;
		featureBias = new int[dim.length - 1];
		for (int j = 0; j < dim.length; ++j) {
			sum += dim[j];
			if (j < dim.length - 1)
				featureBias[j] = sum;
		}
		featureSize = sum;

		if (featureSize > 0) {
			param = new double[rank][featureSize];
			isActive = new boolean[featureSize];
			dFV = new FeatureVector[rank];
			for (int i = 0; i < rank; ++i)
				dFV[i] = new FeatureVector(featureSize);
			
			switch (options.updateMode) {
				case AdaGrad:
					sg = new double[rank][featureSize];
					adaAlpha = options.AdaAlpha;
					adaEps = options.AdaEps;
					break;
				case MIRA:
					total = new double[rank][featureSize];
					C = options.MIRAC;
					break;
				default:
					break;
			}
		}
	}
	
	private void setEmptyFeature() {
		featureSize = 0;
	}
	
	public void constructThreewayStructure() {
		setEmptyFeature();
		setNodeNum(options.learnLabel ? 4 : 3);
		for (int i = 0; i < nodeNum; ++i) {
			node[i] = new ParameterNode(options, pipe, options.R);
			node[i].setNodeNum(0);
		}

		// head 
		int[] dim1 = {1, posNum, posNum, posNum, posNum * posNum, posNum * posNum}; // bias, p_{-1,0,1}, p_{(-1,0),(0,1)}
		node[0].setFeatureSizeAndBias(dim1);
		
		// modifier
		node[1].setFeatureSizeAndBias(dim1);
		
		// direction and distance
		int[] dim2 = {1, 2 * d}; // bias, (direction, distance)
		node[2].setFeatureSizeAndBias(dim2);
		
		if (options.learnLabel) {
			int[] dim3 = {1, labelNum};
			node[3].setFeatureSizeAndBias(dim3); // bias, label
		}
	}
	
	public void constructMultiwayStructure() {
		setEmptyFeature();
		setNodeNum(options.learnLabel ? 6 : 5);
		for (int i = 0; i < nodeNum; ++i) {
			node[i] = new ParameterNode(options, pipe, options.R);
			node[i].setNodeNum(0);
		}

		int[] dim1 = {1, posNum};
		node[0].setFeatureSizeAndBias(dim1);
		node[1].setFeatureSizeAndBias(dim1);
		
		int[] dim2 = {1, posNum, posNum};
		node[2].setFeatureSizeAndBias(dim2);
		node[3].setFeatureSizeAndBias(dim2);
		
		// direction and distance
		int[] dim3 = {1, 2 * d}; // bias, (direction, distance)
		node[4].setFeatureSizeAndBias(dim3);
		
		if (options.learnLabel) {
			int[] dim4 = {1, labelNum};
			node[5].setFeatureSizeAndBias(dim4); // bias, label
		}
	}
	
	public void constructHierarchicalStructure() {
		ParameterNode delexical = null;
		if (options.lexical) {
			setEmptyFeature();
			setNodeNum(2);
			
			// lexical
			node[0] = new ParameterNode(options, pipe, options.R);
			ParameterNode lexical = node[0];
			lexical.setEmptyFeature();
			lexical.setNodeNum(2);
			
			// TODO: add word vector and construct parameters
			if (options.useNN) {
				Utils.ThrowException("not implemented yet");
			}
			else {
				lexical.node[0] = new ParameterNode(options, pipe, options.R * options.extraR); 
				lexical.node[0].setNodeNum(0);
				lexical.node[0].setFeatureSizeAndBias(null);
				
				lexical.node[1] = new ParameterNode(options, pipe, options.R * options.extraR); 
				lexical.node[1].setNodeNum(0);
				lexical.node[1].setFeatureSizeAndBias(null);
			}
			
			// delexical
			node[1] = new ParameterNode(options, pipe, options.R);
			delexical = node[1];
		}
		else {
			delexical = this;
		}

		delexical.setEmptyFeature();
		delexical.setNodeNum(3);
		for (int i = 0; i < 3; ++i)
			delexical.node[i] = new ParameterNode(options, pipe, options.R);

		// head context
		ParameterNode headContext = delexical.node[0];
		headContext.setNodeNum(0);
		int[] dim1 = {1, posNum * pipe.typo.classNum, posNum * pipe.typo.classNum,
				posNum * pipe.typo.familyNum, posNum * pipe.typo.familyNum}; 
		headContext.setFeatureSizeAndBias(dim1);
		
		// modifier context
		ParameterNode modContext = delexical.node[1];
		modContext.setNodeNum(0);
		modContext.setFeatureSizeAndBias(dim1);
		
		if (options.learnLabel) {
			// arc & label
			ParameterNode arc = delexical.node[2];
			int[] dim2 = {pipe.typo.getNumberOfValues(TypoFeatureType.SV) * (2 + 2 * d), 2 * (1 + 2 * d),
					pipe.typo.getNumberOfValues(TypoFeatureType.VO) * (2 + 2 * d), 2 * (1 + 2 * d)};
			arc.setFeatureSizeAndBias(dim2);
			//arc.setEmptyFeature();
			
			arc.setNodeNum(2);
			arc.node[0] = new ParameterNode(options, pipe, options.R);
			arc.node[1] = new ParameterNode(options, pipe, options.R);
			
			// label
			ParameterNode label = arc.node[0];
			int[] dim3 = {1, labelNum};
			label.setFeatureSizeAndBias(dim3); 
			
			// typo
			ParameterNode typo = arc.node[1];
			int[] dim4 = {pipe.typo.getNumberOfValues(TypoFeatureType.Prep) * (2 + 2 * d), 2 * (1 + 2 * d),
					pipe.typo.getNumberOfValues(TypoFeatureType.Gen) * (2 + 2 * d), 2 * (1 + 2 * d),
					pipe.typo.getNumberOfValues(TypoFeatureType.Adj) * (2 + 2 * d), 2 * (1 + 2 * d)};
			typo.setFeatureSizeAndBias(dim4);
			//typo.setEmptyFeature();
			
			typo.setNodeNum(3);
			for (int i = 0; i < 3; ++i)
				typo.node[i] = new ParameterNode(options, pipe, options.R);
			
			// head
			ParameterNode head = typo.node[0];
			head.setNodeNum(0);
			int[] dim5 = {1, posNum};
			head.setFeatureSizeAndBias(dim5);
			
			// modifier
			ParameterNode mod = typo.node[1];
			mod.setNodeNum(0);
			mod.setFeatureSizeAndBias(dim5);
			
			// direction, distance, typo
			ParameterNode dd = typo.node[2];
			int[] dim6 = {1, d * 2 * pipe.typo.classNum, d * 2 * pipe.typo.familyNum};
			dd.setNodeNum(0);
			dd.setFeatureSizeAndBias(dim6);
		}
		else {
			// typo
			ParameterNode typo = delexical.node[2];
			int[] dim4 = {pipe.typo.getNumberOfValues(TypoFeatureType.Prep) * (2 + 2 * d), 2 * (1 + 2 * d),
					pipe.typo.getNumberOfValues(TypoFeatureType.Gen) * (2 + 2 * d), 2 * (1 + 2 * d),
					pipe.typo.getNumberOfValues(TypoFeatureType.Adj) * (2 + 2 * d), 2 * (1 + 2 * d)};
			typo.setFeatureSizeAndBias(dim4);
			
			typo.setNodeNum(3);
			for (int i = 0; i < 3; ++i)
				typo.node[i] = new ParameterNode(options, pipe, options.R);
			
			// head
			ParameterNode head = typo.node[0];
			head.setNodeNum(0);
			int[] dim5 = {1, posNum};
			head.setFeatureSizeAndBias(dim5);
			
			// modifier
			ParameterNode mod = typo.node[1];
			mod.setNodeNum(0);
			mod.setFeatureSizeAndBias(dim5);
			
			// direction, distance, typo
			ParameterNode dd = typo.node[2];
			int[] dim6 = {1, d * 2 * pipe.typo.classNum, d * 2 * pipe.typo.familyNum};
			dd.setNodeNum(0);
			dd.setFeatureSizeAndBias(dim6);
		}
	}
	
	public void setActiveFeature(FeatureVector fv) {
		Utils.Assert(fv.nRows() == featureSize);
		for (int i = 0, L = fv.size(); i < L; ++i)
			isActive[fv.x(i)] = true;
	}
	
	public void randomlyInit(double scale) {
		//if (nodeNum > 0 && featureSize > 0)
		//	scale *= 0.5;
		if (featureSize > 0) {
			int n = 0;
			for (int i = 0; i < featureSize; ++i)
				if (isActive[i])
					n++;
			for (int r = 0; r < rank; ++r) {
				double[] vec = Utils.getRandomVector(n, scale * Math.sqrt(3.0 / n));
				//double[] vec = Utils.getRandomVector(n, scale * (nodeNum > 0 ? 0.5 : 1.0) * Math.sqrt(3.0 / n));
				//double[] vec = Utils.getRandomVector(n, 0.01);
				//double[] vec = Utils.getRandomUnitVector(n);
				int p = 0;
				for (int i = 0; i < featureSize; ++i) {
					if (!isActive[i])
						continue;
					param[r][i] = vec[p];
					p++;
				}
				
				if (options.updateMode == UpdateMode.MIRA) {
					total[r] = param[r].clone();
				}
			}
		}
		
		for (int i = 0; i < nodeNum; ++i) {
			node[i].randomlyInit(scale);
		}
	}
	
	public void updateAda() {
		if (featureSize > 0) {
			for (int r = 0; r < rank; ++r) {
				FeatureVector dfv = dFV[r];
				dfv.aggregate();
				for (int i = 0, L = dfv.size(); i < L; ++i) {
					int x = dfv.x(i);
					double g = dfv.value(i) * (1 - gamma);
					sg[r][x] += g * g;
					param[r][x] += adaAlpha / Math.sqrt(sg[r][x] + adaEps) * g;
				}
				dFV[r].clear();
			}
		}
		for (int i = 0; i < nodeNum; ++i) {
			node[i].updateAda();
		}
	}
	
	public void updateMIRA(double alpha, int updCnt) {
		
		if (featureSize > 0) {
			double lr = Math.min(alpha, C);
			for (int r = 0; r < rank; ++r) {
				FeatureVector dfv = dFV[r];
				dfv.aggregate();
				for (int i = 0, L = dfv.size(); i < L; ++i) {
					int x = dfv.x(i);
					double g = dfv.value(i) * (1 - gamma);
					param[r][x] += lr * g;
					total[r][x] += lr * updCnt * g;
				}
				dFV[r].clear();
			}
		}
		for (int i = 0; i < nodeNum; ++i)
			node[i].updateMIRA(alpha, updCnt);
	}
	
	public double gradientl2Norm() {
		double norm = 0.0;
		if (featureSize > 0) {
			for (int r = 0; r < rank; ++r)
				norm += dFV[r].Squaredl2NormUnsafe();
		}
		for (int i = 0; i < nodeNum; ++i)
			norm += node[i].gradientl2Norm();
		return norm;
	}

	public void averageParameters(int T) {
		if (featureSize > 0) {
			back = param;
			double[][] avg = new double[rank][featureSize];
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < featureSize; ++j) {
					avg[i][j] = (param[i][j] * (T+1) - total[i][j])/T;
				}
			param = avg;
		}
		for (int i = 0; i < nodeNum; ++i)
			node[i].averageParameters(T);
	}
	
	public void unaverageParameters() {
		if (featureSize > 0) {
			param = back;
		}
		for (int i = 0; i < nodeNum; ++i)
			node[i].unaverageParameters();
	}
}
