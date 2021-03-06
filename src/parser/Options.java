package parser;

import java.io.Serializable;
import java.util.Random;

import utils.Utils;

public class Options implements Cloneable, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public enum FeatureMode {
		Basic,			// 1st order arc factored model
		Standard,		// 3rd order using similar features as TurboParser
	}
	
	public enum UpdateMode {
		SGD,
		AdaGrad,
		MIRA,
	}
	
	public enum Dataset {
		CoNLL_UNI,
		CoNLL_07,		// not implemented
	}
	
	public enum TensorMode {
		Threeway,
		Multiway,
		Hierarchical,
		TMultiway,
	}
	
	public UpdateMode updateMode = UpdateMode.AdaGrad;
	public TensorMode tensorMode = TensorMode.Hierarchical;
		
	public String dataDir = "./data";
	public String trainExt = "-universal-train.conll";
	public String testExt = "-universal-test.conll";
	public String devExt = "-universal-dev.conll";
	public boolean train = false;
	public boolean test = false;
	public boolean lexical = false;
	public String wordVectorFile = null;
	public String outFile = "output";
	public String modelFile = "model.out";
	public String typoFile = "typo.txt";
    
	public int maxNumSent = -1;
	public int supSent = 50; 
	public int maxNumIters = 10;
	
	//public LearningMode learningMode = LearningMode.Basic;
	public FeatureMode featureMode = FeatureMode.Standard;
	public boolean projective = false;
	public boolean learnLabel = false;
	public boolean pruning = true;
	public double pruningCoeff = 0.10;
	
	public int numHcThreads = 4;		// hill climbing: number of threads
	
	// Number of hill climbing restarts to converge
	// Training requires more restarts because of cost-augmented decoding
	// Testing is easier therefore needs less restarts
	public int numTrainConverge = 300;	
	public int numTestConverge = 100;	
	
	// for SGD
	public double SGDLearningRate = 0.01;

	// for MIRA
	public boolean MIRAAverage = true;
	public double MIRAC = 1.0;
	
	// for adagrad
	public double AdaAlpha = 0.001;
	public double AdaEps = 1e-5;
	
	// batch
	public int batchSize = 10;		// batch size
	public double lambda = 10.0;	// regularization
	public double tensorLambda = 0.1;
	public boolean useBatch = false;
	
	// tensor
	public double gamma = 1.0;
	public int R = 100;
	public int extraR = 1;
	public boolean useNN = false;
	public boolean initTensorWithPretrain = true;
	
	// feature set
	public boolean useCS = true;		// use consecutive siblings
	public boolean useGP = true;		// use grandparent
	public boolean useHB = true;		// use head bigram
	public boolean useGS = true;		// use grand sibling
	public boolean useTS = true;		// use tri-sibling
	
	public boolean useSupervised = true;
	
	public boolean direct = false;
	public int typoVecDim = 10;
	public String initModel = null;
	
	// CoNLL-UNI languages
//	public enum PossibleLang {
//		English,
//		French,
//		German,
//		Indonesian,
//		Italian,
//		Japanese,
//		Korean,
//		Portuguese,
//		Spanish,
//		Swedish,
//	}
//	PossibleLang lang;
//	
	public String langString[] = {"en", "fr", "de", "id", "it", "ja",
			"ko", "pt-br", "es", "sv"};

	public String sub1LangString[] = {"en", "fr", "de", "id", "it", 
			"ko", "pt-br", "es", "sv"};
	public String sub2LangString[] = {"en", "fr", "de", "id", "it", 
			"pt-br", "es", "sv"};
	public String ud1LangString[] = {"cs", "de", "en", "es", "fi", "fr", "ga", "hu", "it", "sv"};

	public String euroLangString[] = {"en", "fr", "de", "it", "pt-br", "es", "sv"};
	
	
	public String targetLangStr = "";
	public int targetLang;
	public Dataset dataset = Dataset.CoNLL_UNI;

	public int seed = 0;
	
	public Options() {
		
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		return super.clone();
	}
	
    public void processArguments(String[] args) {
    	
    	for (String arg : args) {
    		if (arg.equals("train")) {
    			train = true;
    		}
    		else if (arg.equals("test")) {
    			test = true;
    		}
    		else if (arg.startsWith("data:")) {
    			dataDir = arg.split(":")[1];
    		}
    		else if (arg.startsWith("train-ext:")) {
    			trainExt = arg.split(":")[1];
    		}
    		else if (arg.startsWith("test-ext:")) {
    			testExt = arg.split(":")[1];
    		}
    		else if (arg.startsWith("output-file:")) {
    			outFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("model-file:")) {
    			modelFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("init-model:")) {
    			initModel = arg.split(":")[1];
    		}
    		else if (arg.startsWith("typo-file:")) {
    			typoFile = arg.split(":")[1];
    		}
            else if (arg.startsWith("update:")) {
            	String str = arg.split(":")[1];
            	UpdateMode[] values = UpdateMode.values();
                for (int i = 0; i < values.length; ++i)
                	if (str.equalsIgnoreCase(values[i].name())) {
                		updateMode = values[i];
                		break;
                	}
            }
            else if (arg.startsWith("tensor:")) {
            	String str = arg.split(":")[1];
            	TensorMode[] values = TensorMode.values();
                for (int i = 0; i < values.length; ++i)
                	if (str.equalsIgnoreCase(values[i].name())) {
                		tensorMode = values[i];
                		break;
                	}
            }
            else if (arg.startsWith("max-sent:")) {
                maxNumSent = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("sup-sent:")) {
                supSent = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("iters:")) {
                maxNumIters = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("feature:")) {
            	String str = arg.split(":")[1];
            	if (str.equals("basic"))
            		featureMode = FeatureMode.Basic;
            	else if (str.equals("standard"))
            		featureMode = FeatureMode.Standard;
            }
    		else if (arg.startsWith("label")) {
    			learnLabel = Boolean.parseBoolean(arg.split(":")[1]);
    		}
    		else if (arg.startsWith("lexical")) {
    			lexical = Boolean.parseBoolean(arg.split(":")[1]);
    		}
    		else if (arg.startsWith("pretrain")) {
    			initTensorWithPretrain = Boolean.parseBoolean(arg.split(":")[1]);
    		}
    		else if (arg.startsWith("word-vector:")) {
            	wordVectorFile = arg.split(":")[1];
            }
            else if (arg.startsWith("proj")) {
                projective = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("lr:")) {
            	SGDLearningRate = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("average:")) {
            	MIRAAverage = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("C:")) {
            	MIRAC = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("adalr:")) {
            	AdaAlpha = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("gamma:")) {
            	gamma = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("R:")) {
                R = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("extra-R:")) {
                extraR = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("nn:")) {
                useNN = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("batch:")) {
                useBatch = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("direct:")) {
                direct = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("pruning:")) {
                pruning = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("pruning-weight:")) {
            	pruningCoeff = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("thread:")) {
            	numHcThreads = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("converge:")) {
            	numTrainConverge = Integer.parseInt(arg.split(":")[1]);
            	numTestConverge = numTrainConverge;
            }
            else if (arg.startsWith("converge-train:")) {
            	numTrainConverge = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("converge-test:")) {
            	numTestConverge = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("target:")) {
            	targetLangStr = arg.split(":")[1];
            }
            else if (arg.startsWith("seed")) {
            	seed = Integer.parseInt(arg.split(":")[1]);
            }
    	}    	
        
    	switch (featureMode) {
    		case Basic:
    			useCS = false;
    			useGP = false;
    			useHB = false;
    			useGS = false;
    			useTS = false;
    			
    			pruning = false;
    			break;
    		case Standard:
    			break;
    		default:
    			break;
    	}
    	
    	langString = ud1LangString;
    	if (typoFile.contains("euro"))
    		langString = euroLangString;
    	else if (typoFile.contains("sub1"))
    		langString = sub1LangString;
    	else if (typoFile.contains("sub2"))
    		langString = sub2LangString;
    	targetLang = findLang(targetLangStr);
    	
    	Utils.rnd = new Random(seed);
    }
    
    public void printOptions() {
    	System.out.println("------\nFLAGS\n------");
    	System.out.println("train-ext: " + trainExt);
    	System.out.println("test-ext: " + testExt);
    	System.out.println("model-name: " + modelFile);
        System.out.println("output-file: " + outFile);
        System.out.println("typo-file: " + typoFile);
    	System.out.println("train: " + train);
    	System.out.println("test: " + test);
        System.out.println("iters: " + maxNumIters);
        System.out.println("lexical: " + lexical);
    	System.out.println("label: " + learnLabel);
        System.out.println("max-sent: " + maxNumSent);  
        System.out.println("seed: " + seed);  
        System.out.println("tensor mode:" + tensorMode.name());
        System.out.println("update mode:" + updateMode.name());
        if (updateMode == UpdateMode.SGD) {
        	System.out.println("learning rate: " + SGDLearningRate);
        }
        else if (updateMode == UpdateMode.MIRA) {
        	System.out.println("C: " + MIRAC);
        }
        else if (updateMode == UpdateMode.AdaGrad) {
        	System.out.println("Ada alphs: " + AdaAlpha);
        }
        
        System.out.println("gamma: " + gamma);
        System.out.println("R: " + R);
        System.out.println("extra-R: " + extraR);
        System.out.println("use NN: " + useNN);
        System.out.println("use batch: " + useBatch);
        System.out.println("direct transfer: " + direct);
        System.out.println("word-vector:" + wordVectorFile);
        System.out.println("projective: " + projective);
        System.out.println("pruning: " + pruning);
        System.out.println("hill-climbing converge (train): " + numTrainConverge);
        System.out.println("hill-climbing converge (test): " + numTestConverge);
        System.out.println("thread: " + numHcThreads);
        System.out.println("dataset: " + dataset.name());
        System.out.println("target language: " + targetLangStr);
        
        System.out.println();
        System.out.println("use consecutive siblings: " + useCS);
        System.out.println("use grandparent: " + useGP);
        System.out.println("use head bigram: " + useHB);
        System.out.println("use grand siblings: " + useGS);
        System.out.println("use tri-siblings: " + useTS);
        System.out.println("model: " + featureMode.name());

    	System.out.println("------\n");
    }
    
    int findLang(String langStr) {
    	for (int i = 0; i < langString.length; ++i)
    		if (langStr.equals(langString[i])) {
    			return i;
    		}
    	Utils.ThrowException("Error: unknow language");
    	return -1;
    }

}
