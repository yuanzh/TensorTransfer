package parser;

import java.io.IOException;

import parser.Options.UpdateMode;
import parser.decoding.DependencyDecoder;
import parser.io.DependencyReader;
import parser.tensor.ParameterNode;
import utils.Utils;

public class TensorTransfer {

	public Options options;
	public DependencyPipe pipe;
	public Parameters parameters;
	
    public void train(DependencyInstance[] lstTrain) throws IOException, CloneNotSupportedException {
    	long start = 0, end = 0;
    	
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
        			if (pred.heads[m] == inst.heads[m] && 
        				(!options.learnLabel || pred.deplbids[m] == inst.deplbids[m]))
        				corr++;
    		    }
    		    
        		if (corr != n - 1) {
        			loss += parameters.update(inst, pred, fd, iIter * N + i + 1);
                }

        		acc += corr;
        		tot += n - 1;        		
        		
    		}
    		System.out.printf("%n  Iter %d\tloss=%.4f\tacc=%.4f\t[%ds]%n", iIter+1, loss, acc/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		
    		
    		// evaluate on a development set
    		if (evalAndSave) {		
                if (options.updateMode == UpdateMode.MIRA && options.MIRAAverage) 
                	parameters.averageParameters((iIter+1)*N);
    			
    			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			int target = options.targetLang;
	  			System.out.printf(" Evaluation: %s%n", options.langString[target]);
	  			System.out.println(); 
	  			double res = evaluateSet(target, false, false);
	  			//double res = 0.0;
	  			if (res > bestDevAcc) {
	  				bestDevAcc = res;
	  				//saveModel(options.modelFile);
	  			}
	  			evaluateSet(target, false, true);

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
    	

    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);   	
    	
    	Evaluator eval = new Evaluator(options, pipe);
    	
		long start = System.currentTimeMillis();
    	
    	DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		FeatureData fd = new FeatureData(inst, this, true, false);
            DependencyInstance predInst = decoder.decode(inst, fd);
            eval.add(inst, predInst, evalWithPunc);
    		
    		inst = pipe.createInstance(reader);
    	}
    	
    	reader.close();
    	
    	System.out.printf("  Tokens: %d%n", eval.tot);
    	System.out.printf("  Sentences: %d%n", eval.nsents);
    	System.out.printf("  UAS=%.6f\tLAS=%.6f\tCAS=%.6f\t[%.2fs]%n",
    			eval.UAS(), eval.LAS(), eval.CAS(),
    			(System.currentTimeMillis() - start)/1000.0);

    	decoder.shutdown();

        return eval.UAS();
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
//			parser.saveModel();
		}
	}

}
