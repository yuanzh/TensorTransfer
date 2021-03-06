package parser;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import parser.Options.FeatureMode;
import parser.Options.TensorMode;
import parser.Options.UpdateMode;
import parser.decoding.DependencyDecoder;
import parser.feature.FeatureRepo;
import parser.io.DependencyReader;
import parser.io.DependencyWriter;
import parser.tensor.LowRankParam;
import parser.tensor.ParameterNode;
import utils.TypologicalInfo;
import utils.Utils;
import utils.WordVector;

public class TensorTransfer {

	public Options options;
	public DependencyPipe pipe;
	public Parameters parameters;
	
    public void train(DependencyInstance[] lstTrain) throws IOException, CloneNotSupportedException, ClassNotFoundException {
    	long start = 0, end = 0;
    	
    	if (options.initModel != null) {
    		System.out.println("init model");
            ObjectInputStream in = new ObjectInputStream(
                    new GZIPInputStream(new FileInputStream(options.initModel)));    
            in.readObject();		// DependencyPIpe
            Parameters param = (Parameters) in.readObject();
            in.close();
    		
            Utils.Assert(parameters.size == param.size);
            //System.arraycopy(param.params, 0, parameters.params, 0, param.params.length);
            //if (options.updateMode == UpdateMode.MIRA) {
            //	 System.arraycopy(param.params, 0, parameters.total, 0, param.params.length);
            //}
            parameters.reg = param.params;
            
            // TODO: copy tensor parameters
            if (options.gamma < 1.0) {
            	
            }
    	}
    	else if (options.R > 0 && options.gamma < 1 && options.initTensorWithPretrain) {

        	Options optionsBak = (Options) options.clone();
        	options.featureMode= FeatureMode.Basic;
        	options.R = 0;
        	options.gamma = 1.0;
        	options.maxNumIters = 1;
        	options.AdaAlpha = 0.01;
        	//options.MIRAC = 0.01;
        	parameters.gamma = 1.0;
        	parameters.adaAlpha = 0.01;
        	//parameters.C = 0.01;
        	parameters.pn.setGamma(1.0);
        	parameters.pn.setAdaAlpha(0.01);
        	//parameters.pn.setMIRAC(0.01);
    		System.out.println("=============================================");
    		System.out.printf(" Pre-training:%n");
    		System.out.println("=============================================");
    		
    		start = System.currentTimeMillis();

    		System.out.println("Running " + options.updateMode.name() + " ... ");
    		trainIter(lstTrain, true);
    		System.out.println();
    		
    		System.out.println("Init tensor ... ");
    		LowRankParam tensor = new LowRankParam(parameters.pn);
    		if (options.tensorMode == TensorMode.Threeway) {
    			pipe.ff.fillThreewayParameters(tensor, parameters);
        		tensor.decomposeThreeway();
    		}
    		else if (options.tensorMode == TensorMode.Multiway) {
    			pipe.ff.fillMultiwayParameters(tensor, parameters);
    			tensor.decomposeMultiway();
    		}
    		else if (options.tensorMode == TensorMode.Hierarchical) {
    			pipe.ff.fillHierarchichalParameters(tensor, parameters);
    			tensor.decomposeHierarchicalway();
    		}
    		else if (options.tensorMode == TensorMode.TMultiway) {
    			pipe.ff.fillTMultiwayParameters(tensor, parameters);
    			tensor.decomposeTMultiway();
    		}
    		else
    			Utils.ThrowException("not implemented yet");
    		
            System.out.println();
    		end = System.currentTimeMillis();
    		
    		options.featureMode = optionsBak.featureMode;
    		options.R = optionsBak.R;
    		options.gamma = optionsBak.gamma;
    		options.maxNumIters = optionsBak.maxNumIters;
    		options.AdaAlpha = optionsBak.AdaAlpha;
    		//options.MIRAC = optionsBak.MIRAC;
    		parameters.gamma = optionsBak.gamma;
    		parameters.adaAlpha = optionsBak.AdaAlpha;
    		//parameters.C = optionsBak.MIRAC;
    		parameters.pn.setGamma(optionsBak.gamma);
    		parameters.pn.setAdaAlpha(optionsBak.AdaAlpha);
    		//parameters.pn.setMIRAC(optionsBak.MIRAC);
    		parameters.clearTheta();
            System.out.println();
            System.out.printf("Pre-training took %d ms.%n", end-start);    		
    		System.out.println("=============================================");
    		System.out.println();	    

        }

        System.out.println("=============================================");
		System.out.printf(" Training:%n");
		System.out.println("=============================================");
		
		start = System.currentTimeMillis();

		System.out.println("Running " + options.updateMode.name() + " ... ");
		trainIter(lstTrain, true);
		System.out.println();
		
		end = System.currentTimeMillis();
		
		System.out.printf("Training took %d ms.%n", end-start);    		
		System.out.println("=============================================");
		System.out.println();		    	
    }

    public void trainIter(DependencyInstance[] lstTrain, boolean evalAndSave) throws IOException
    {
    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);
    	
    	int N = lstTrain.length;
    	//int printPeriod = 10000 < N ? N/10 : 1000;
    	int printPeriod = 1000;

		double bestDevAcc = 0.0;

    	for (int iIter = 0; iIter < options.maxNumIters; ++iIter) {
    	    
    		long start = 0;
    		
    		int acc = 0, tot = 0;
    		double loss = 0.0;
    		start = System.currentTimeMillis();
                		  
    		int b = 0;
    		for (int i = 0; i < N; ++i) {
    			
    			if ((i + 1) % printPeriod == 0) {
					System.out.printf("  %d (time=%ds)", (i+1),
						(System.currentTimeMillis()-start)/1000);
					//System.out.println(parameters.nnW.tForward + " " + parameters.nnW.tGradient + " " + parameters.nnW.tNorm + " " + parameters.nnW.tUpdate);
    			}

    			DependencyInstance inst = lstTrain[i];
       		    int n = inst.length;
       		 
    			FeatureData fd = new FeatureData(inst, this, true, true);
    		    
    		    DependencyInstance pred = decoder.decode(inst, fd);
    		    
        		int corr = 0;
    		    for (int m = 1; m < n; ++m) {
        			if (pred.heads[m] == inst.heads[m])
        				corr++;
    		    }
    		    
        		if (corr != n - 1) {
        			if (!options.useBatch)
        				loss += parameters.update(inst, pred, fd);
        			else
        				loss += parameters.addGradient(inst, pred, fd);
                }

        		acc += corr;
        		tot += n - 1;    
        		
        		// predict label
        		if (options.learnLabel) {
        			pred.heads = inst.heads;
        			fd.predictLabels(pred.heads, pred.deplbids);
        			int la = 0;
        		    for (int m = 1; m < n; ++m) {
            			if (pred.deplbids[m] == inst.deplbids[m])
            				la++;
        		    }
        			if (la != n-1) {
        				if (!options.useBatch)
        					loss += parameters.updateLabel(inst, pred, fd);
        				else
        					loss += parameters.addLabelGradient(inst, pred, fd);
        			}
        		}
        		
        		b++;
        		if (b == options.batchSize) {
        			//System.out.println("aaa");
        			if (options.useBatch)
        				parameters.batchUpdate(N);
        			b = 0;
        		}
    		}
    		System.out.printf("%n  Iter %d\tloss=%.4f\tacc=%.4f\t[%ds]%n", iIter+1, loss, acc/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		
    		parameters.printNorm();
    		
    		
    		// evaluate on a development set
    		if (evalAndSave) {		
                if (options.updateMode == UpdateMode.MIRA && options.MIRAAverage) 
                	parameters.averageParameters();
                if ((iIter + 1) % 5 == 0) {
                	saveModel(options.modelFile + "." + iIter);
                }
                
                //double avgDev = 0.0;
                //double avgTest = 0.0;
                //for (int lang = 0; lang < pipe.typo.langNum; ++lang) {
    			
	    			System.out.println();
		  			System.out.println("_____________________________________________");
		  			System.out.println();
		  			int target = options.targetLang;
		  			//int target = lang;
		  			System.out.printf(" Evaluation: %s%n", options.langString[target]);
		  			System.out.println(); 
		  			double res = evaluateSet(target, false, false);
		  			//avgTest += res;
		  			//double res = 0.0;
		  			if (res > bestDevAcc) {
		  				bestDevAcc = res;
		  				//saveModel(options.modelFile);
		  			}
		  			evaluateSet(target, false, true);
                //}
                //System.out.println(avgTest / pipe.typo.langNum + " " + avgDev / pipe.typo.langNum);

                if (options.updateMode == UpdateMode.MIRA && options.MIRAAverage) 
                	parameters.unaverageParameters();
    		} 
    	}
    	
    }

    public double evaluateSet(int target, boolean evalWithPunc, boolean isDev)
    		throws IOException {
    	DependencyReader reader = DependencyReader.createDependencyReader(options, target);
    	if (isDev)
        	reader.startReading(pipe.constructDevFileName(target));
    	else
    		reader.startReading(pipe.constructTestFileName(target));
    	
    	DependencyWriter writer = null;
    	if (isDev && options.outFile != null) {
    		writer = DependencyWriter.createDependencyWriter(options, target, pipe);
    		writer.startWriting(options.outFile);
    	}

    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);   	
    	
    	Evaluator eval = new Evaluator(options, pipe);
    	
		long start = System.currentTimeMillis();
    	
    	DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		FeatureData fd = new FeatureData(inst, this, true, false);
            DependencyInstance predInst = decoder.decode(inst, fd);
            if (options.learnLabel) {
            	fd.predictLabels(predInst.heads, predInst.deplbids);
            }
            
            eval.add(inst, predInst, evalWithPunc);
    		
    		if (writer != null) {
    			inst.heads = predInst.heads;
    			inst.deplbids = predInst.deplbids;
    			writer.writeInstance(inst);
    		}

    		inst = pipe.createInstance(reader);
    	}
    	
    	reader.close();
    	if (writer != null) writer.close();
    	
    	System.out.printf("  Tokens: %d%n", eval.tot);
    	System.out.printf("  Sentences: %d%n", eval.nsents);
    	System.out.printf("  UAS=%.6f\tLAS=%.6f\tCAS=%.6f\t[%.2fs]%n",
    			eval.UAS(), eval.LAS(), eval.CAS(),
    			(System.currentTimeMillis() - start)/1000.0);

    	decoder.shutdown();

        return eval.UAS();
    }
    
    public void saveModel(String file) throws IOException 
    {
    	System.out.println("save model to " + file);
    	ObjectOutputStream out = new ObjectOutputStream(
    			new GZIPOutputStream(new FileOutputStream(file)));
    	out.writeObject(pipe);
    	out.writeObject(parameters);
    	out.writeObject(options);
    	//if (options.pruning && options.learningMode != LearningMode.Basic) 
    	//	out.writeObject(pruner);
    	out.close();
    }
	
    public void loadModel(String file) throws IOException, ClassNotFoundException 
    {
    	System.out.println("load model from " + file);
        ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(file)));    
        pipe = (DependencyPipe) in.readObject();
        parameters = (Parameters) in.readObject();
        options = (Options) in.readObject();
        //if (options.pruning && options.learningMode != LearningMode.Basic)
        	//pruner = (DependencyParser) in.readObject();
        //	pruner = (BasicArcPruner) in.readObject();
        pipe.options = options;
        parameters.options = options;        
        
		TypologicalInfo typo = new TypologicalInfo(options);
		pipe.typo = typo;
        pipe.ff.typo = typo;
		FeatureRepo fr = new FeatureRepo(options, pipe.ff);
		pipe.fr = fr;
		pipe.ff.fr = fr;
		
		if (options.lexical) {
			WordVector wv = new WordVector(options);
			pipe.wv = wv;
			pipe.ff.wv = wv;
			pipe.dictionaries.wv = wv;
		}
       
        in.close();
        pipe.closeAlphabets();
    }
    
    public void outputWeight() {
    	ParameterNode pn = parameters.pn;
		ParameterNode hcpn = pn.node[2];
		ParameterNode mcpn = pn.node[3];
		ParameterNode lpn = pn.node[6];
		ParameterNode tpn = pn.node[4];
		ParameterNode hpn = pn.node[0];
		ParameterNode mpn = pn.node[1];
		ParameterNode dpn = pn.node[5];
		
		// Adj-Noun + left arc + verb + noun
		double[] hcw = getCol(hcpn.param, 0);
		double[] mcw = getCol(mcpn.param, 0);
		double[] lw = getCol(lpn.param, 0);
		double[] tw = getCol(tpn.param, tpn.featureBias[7]);
		double[] hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_VERB);
		double[] mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		double[] dw = getCol(dpn.param, dpn.featureBias[3]);
		double w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adj-Noun + left arc + verb + noun: " + w);
		
		// Adj-Noun + left arc + noun + adj
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, 0);
		tw = getCol(tpn.param, tpn.featureBias[7]);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_NOUN);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_ADJ);
		dw = getCol(dpn.param, dpn.featureBias[3]);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adj-Noun + left arc + noun + adj: " + w);
		
		// Adj-Noun + right arc + adp + noun
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, 0);
		tw = getCol(tpn.param, tpn.featureBias[7]);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_ADP);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		dw = getCol(dpn.param, dpn.featureBias[3] + 5);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adj-Noun + right arc + noun + adp: " + w);
		
                // Adj-Noun + left arc + verb + noun
                hcw = getCol(hcpn.param, 0);
                mcw = getCol(mcpn.param, 0);
                lw = getCol(lpn.param, 0);
                tw = getCol(tpn.param, tpn.featureBias[7]);
                hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_VERB);
                mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_PRON);
                dw = getCol(dpn.param, dpn.featureBias[3]);
                w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
                System.out.println("Adj-Noun + left arc + verb + pron: " + w);

                // Adj-Noun + right arc + noun + noun
                hcw = getCol(hcpn.param, 0);
                mcw = getCol(mcpn.param, 0);
                lw = getCol(lpn.param, 0);
                tw = getCol(tpn.param, tpn.featureBias[7]);
                hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_NOUN);
                mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
                dw = getCol(dpn.param, dpn.featureBias[3]);
                w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
                System.out.println("Adj-Noun + left arc + noun + noun: " + w);

		// Subj-Verb + left arc + verb + noun
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, lpn.featureBias[0] + pipe.ff.LABEL_SBJ);
		tw = getCol(tpn.param, tpn.featureBias[0]);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_VERB);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		dw = getCol(dpn.param, dpn.featureBias[3]);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Subj-Verb + left arc + verb + noun: " + w);
		
		// Subj-Verb + left arc + noun + adj
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, lpn.featureBias[0] + pipe.ff.LABEL_SBJ);
		tw = getCol(tpn.param, tpn.featureBias[0]);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_NOUN);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_ADJ);
		dw = getCol(dpn.param, dpn.featureBias[3]);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Subj-Verb + left arc + noun + adj: " + w);
		
		// Subj-Verb + right arc + adp + noun
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, lpn.featureBias[0] + pipe.ff.LABEL_SBJ);
		tw = getCol(tpn.param, tpn.featureBias[0]);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_ADP);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		dw = getCol(dpn.param, dpn.featureBias[3] + 5);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Subj-Verb + left arc + adp + noun: " + w);
		
		// Adp-Noun + left arc + verb + noun
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, 0);
		tw = getCol(tpn.param, tpn.featureBias[4] + 1);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_VERB);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		dw = getCol(dpn.param, dpn.featureBias[3]);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adp-Noun + left arc + verb + noun: " + w);
		
		// Adp-Noun + left arc + noun + adj
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, 0);
		tw = getCol(tpn.param, tpn.featureBias[4] + 1);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_NOUN);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_ADJ);
		dw = getCol(dpn.param, dpn.featureBias[3]);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adp-Noun + left arc + noun + adj: " + w);
		
		// Adp-Noun + right arc + adp + noun
		hcw = getCol(hcpn.param, 0);
		mcw = getCol(mcpn.param, 0);
		lw = getCol(lpn.param, 0);
		tw = getCol(tpn.param, tpn.featureBias[4] + 1);
		hw = getCol(hpn.param, hpn.featureBias[0] + pipe.ff.POS_ADP);
		mw = getCol(mpn.param, mpn.featureBias[0] + pipe.ff.POS_NOUN);
		dw = getCol(dpn.param, dpn.featureBias[3] + 5);
		w = Utils.sum(Utils.dot(hcw, mcw, lw, tw, hw, mw, dw));
		System.out.println("Adp-Noun + right arc + adp + noun: " + w);
   }
    
    public double[] getCol(double[][] a, int col) {
    	double[] ret = new double[a.length];
    	for (int i = 0; i < ret.length; ++i)
    		ret[i] = a[i][col];
    	return ret;
    }
    
    /**
	 * @param args
	 */
	public static void main(String[] args)  
			throws IOException, ClassNotFoundException, CloneNotSupportedException {
		Options options = new Options();
		options.processArguments(args);		

		if (options.train) {
			TensorTransfer parser = new TensorTransfer();
			parser.options = options;
			options.printOptions();
			
			DependencyPipe pipe = new DependencyPipe(options);
			parser.pipe = pipe;
			pipe.createDictionaries();
			
			ParameterNode pn = new ParameterNode(options, pipe, options.R);
			pn.constructStructure();
			
			pipe.createAlphabets(pn);
			pn.randomlyInit(1.0);
			
			DependencyInstance[] lstTrain = pipe.createInstances();

			Parameters parameters = new Parameters(options, pipe, pn);
			parser.parameters = parameters;
			
			parser.train(lstTrain);
//			if (options.dev && options.learningMode != LearningMode.Basic) 
//				parser.tuneSpeed();
            if (options.updateMode == UpdateMode.MIRA && options.MIRAAverage) 
            	parser.parameters.averageParameters();
			parser.saveModel(options.modelFile + ".last");
		}
		
		if (options.test) {
			TensorTransfer parser = new TensorTransfer();
			parser.options = options;			
			
			parser.loadModel(options.modelFile + ".9");
			parser.options.processArguments(args);
			parser.options.printOptions(); 
			
			//parser.outputWeight();
			
  			//int target = options.targetLang;
  			//System.out.printf(" Evaluation: %s%n", options.langString[target]);
  			//System.out.println(); 
  			//parser.evaluateSet(target, false, false);
  			//parser.evaluateSet(target, false, true);
		}
		
	}

}
