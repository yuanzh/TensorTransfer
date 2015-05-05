package parser;

import static utils.DictionarySet.DictionaryTypes.DEPLABEL;
import static utils.DictionarySet.DictionaryTypes.POS;

import java.io.*;
import java.util.*;

import parser.Options.Dataset;
import parser.feature.FeatureFactory;
import parser.feature.FeatureRepo;
import parser.io.DependencyReader;
import parser.tensor.ParameterNode;

import utils.Dictionary;
import utils.DictionarySet;
import utils.TypologicalInfo;
import utils.Utils;

public class DependencyPipe implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

    public Options options;
    public DictionarySet dictionaries;
    public FeatureFactory ff;
    public transient TypologicalInfo typo;
    public transient FeatureRepo fr;
        
    public String[] types;					// array that maps label index to label string
    public String[] poses;

    public DependencyPipe(Options options) throws IOException 
	{
		dictionaries = new DictionarySet();
		ff = new FeatureFactory(options);
		typo = new TypologicalInfo(options);
		
		ff.typo = typo;
		
		this.options = options;
	}
	
	public String constructTrainFileName(int l) {
		if (options.dataset == Dataset.CoNLL_UNI) {
			return options.dataDir + "/universal_treebanks_v2.0/std/" + options.langString[l] + "/" + options.langString[l] + options.trainExt;
		}
		else {
			Utils.ThrowException("not implemented yet");
			return null;
		}
	}
	
	public String constructTestFileName(int l) {
		if (options.dataset == Dataset.CoNLL_UNI) {
			return options.dataDir + "/universal_treebanks_v2.0/std/" + options.langString[l] + "/" + options.langString[l] + options.testExt;
		}
		else {
			Utils.ThrowException("not implemented yet");
			return null;
		}
	}
	
	public String constructDevFileName(int l) {
		if (options.dataset == Dataset.CoNLL_UNI) {
			return options.dataDir + "/universal_treebanks_v2.0/std/" + options.langString[l] + "/" + options.langString[l] + options.devExt;
		}
		else {
			Utils.ThrowException("not implemented yet");
			return null;
		}
	}
	
	public void createDictionaries() throws IOException 
	{
		long start = System.currentTimeMillis();
		System.out.print("Creating dictionaries ... ");
		
        dictionaries.setCounters();
        
		int cnt = 0;
        for (int l = 0; l < options.langString.length; ++l) {
        	if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        		continue;
        	
        	String file = constructTrainFileName(l);
        	System.out.print(" " + options.langString[l] + " ");
        	
    		DependencyReader reader = DependencyReader.createDependencyReader(options, l);
    		reader.startReading(file);
    		DependencyInstance inst = reader.nextInstance();
    		
    		while (inst != null) {
    			inst.setInstIds(dictionaries);
    			
    			inst = reader.nextInstance();	
    			++cnt;
    			//if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;
    		}
    		reader.close();
    		
    		//if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;
        }
        System.out.println("Done.");
        
		dictionaries.filterDictionary(DEPLABEL);
		dictionaries.closeCounters();
		
		ff.TOKEN_START = dictionaries.lookupIndex(POS, "#TOKEN_START#") - 1;
		ff.TOKEN_END = dictionaries.lookupIndex(POS, "#TOKEN_END#") - 1;
		ff.TOKEN_MID = dictionaries.lookupIndex(POS, "#TOKEN_MID#") - 1;
		
		ff.POS_NOUN = dictionaries.lookupIndex(POS, "NOUN") - 1;
		ff.POS_PRON = dictionaries.lookupIndex(POS, "PRON") - 1;
		ff.POS_ADJ = dictionaries.lookupIndex(POS, "ADJ") - 1;
		ff.POS_VERB = dictionaries.lookupIndex(POS, "VERB") - 1;
		ff.POS_ADP = dictionaries.lookupIndex(POS, "ADP") - 1;
		
		ff.LABEL_SBJ = dictionaries.lookupIndex(DEPLABEL, "nsubj") - 1;
		ff.LABEL_SBJPASS = dictionaries.lookupIndex(DEPLABEL, "nsubjpass") - 1;
		ff.LABEL_DOBJ = dictionaries.lookupIndex(DEPLABEL, "dobj") - 1;
		ff.LABEL_IOBJ = dictionaries.lookupIndex(DEPLABEL, "iobj") - 1;
        
		dictionaries.stopGrowth(DEPLABEL);
		dictionaries.stopGrowth(POS);
				
		ff.tagNumBits = Math.max(Utils.log2(dictionaries.size(POS) + 1), typo.bit + 1);
		ff.depNumBits = Utils.log2(dictionaries.size(DEPLABEL)*2 + 1);
		
		if (options.learnLabel)
			ff.flagBits = ff.depNumBits + 4;
		else
			ff.flagBits = 4;
		
		ff.posNum = dictionaries.size(POS);
		ff.labelNum = dictionaries.size(DEPLABEL);
		
		types = new String[dictionaries.size(DEPLABEL)];	 
		Dictionary labelDict = dictionaries.get(DEPLABEL);
		Object[] keys = labelDict.toArray();
		for (int i = 0; i < keys.length; ++i) {
			int id = labelDict.lookupIndex((String)keys[i]);
			types[id-1] = (String)keys[i];
			//System.out.println(types[id - 1]);
		}
		
		poses = new String[dictionaries.size(POS)];	 
		Dictionary posDict = dictionaries.get(POS);
		keys = posDict.toArray();
		for (int i = 0; i < keys.length; ++i) {
			int id = posDict.lookupIndex((String)keys[i]);
			poses[id - 1] = (String)keys[i];
			//System.out.println(poses[id - 1]);
		}
		
		System.out.println("NOUN: " + poses[ff.POS_NOUN] + " " + ff.POS_NOUN);
		System.out.println("PRON: " + poses[ff.POS_PRON] + " " + ff.POS_PRON);
		System.out.println("VERB: " + poses[ff.POS_VERB] + " " + ff.POS_VERB);
		System.out.println("ADJ: " + poses[ff.POS_ADJ] + " " + ff.POS_ADJ);
		System.out.println("ADP: " + poses[ff.POS_ADP] + " " + ff.POS_ADP);
		System.out.println("nsubj: " + types[ff.LABEL_SBJ] + " " + ff.LABEL_SBJ);
		System.out.println("nsubjpass: " + types[ff.LABEL_SBJPASS] + " " + ff.LABEL_SBJPASS);
		System.out.println("dobj: " + types[ff.LABEL_DOBJ] + " " + ff.LABEL_DOBJ);
		System.out.println("iobj: " + types[ff.LABEL_IOBJ] + " " + ff.LABEL_IOBJ);

		System.out.printf("Tag/label items: %d %d %d (%d bits)  %d (%d bits)%n", 
				dictionaries.size(POS), typo.familyNum, typo.classNum, ff.tagNumBits,
				dictionaries.size(DEPLABEL), ff.depNumBits);
		System.out.printf("Flag Bits: %d%n", ff.flagBits);
		System.out.printf("Creation took [%d ms]%n", System.currentTimeMillis() - start);
	}

	public void createAlphabets(ParameterNode pn) throws IOException 
	{
		ff.pn = pn;
	    
		fr = new FeatureRepo(options, ff);
		ff.fr = fr;
		
		long start = System.currentTimeMillis();
		System.out.print("Creating Alphabet ... ");
		
		HashSet<String> posTagSet = new HashSet<String>();
		int cnt = 0;
        for (int l = 0; l < options.langString.length; ++l) {
        	if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        		continue;
        	
        	String file = constructTrainFileName(l);
        	System.out.print(" " + options.langString[l] + " ");
            
    		DependencyReader reader = DependencyReader.createDependencyReader(options, l);
    		reader.startReading(file);
    		
    		DependencyInstance inst = reader.nextInstance();
    		
    		while(inst != null) {
    			
    			for (int i = 0; i < inst.length; ++i) {
    				if (inst.postags != null) posTagSet.add(inst.postags[i]);
    			}
    			
    			inst.setInstIds(dictionaries);
    			
    		    ff.initFeatureAlphabets(inst);
    				
    		    inst = reader.nextInstance();
    		    cnt++;
    	        //if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;
    		}
    		reader.close();
    		//if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;		
        }
        System.out.println("Done.");
		System.out.printf("[%d ms]%n", System.currentTimeMillis() - start);
		
		closeAlphabets();

		System.out.printf("Num of CONLL fine POS tags: %d %d%n", posTagSet.size(), dictionaries.size(POS));
		System.out.printf("Num of labels: %d%n", types.length);
		System.out.printf("Num of Syntactic Features: %d%n", ff.featureHashSet.size());
	}

    public void closeAlphabets() 
    {
		ff.closeAlphabets();
    }

    public DependencyInstance[] createInstances() throws IOException 
    {
    	
    	long start = System.currentTimeMillis();
    	System.out.print("Creating instances ... ");
    	
		ArrayList<DependencyInstance> lt = new ArrayList<DependencyInstance>();
		int cnt = 0;
        for (int l = 0; l < options.langString.length; ++l) {
        	if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        		continue;
        	
        	String file = constructTrainFileName(l);
        	System.out.print(" " + options.langString[l] + " ");
            
        	DependencyReader reader = DependencyReader.createDependencyReader(options, l);
    		reader.startReading(file);

    		DependencyInstance inst = reader.nextInstance();
    						
    		while(inst != null) {
    			
    			inst.setInstIds(dictionaries);
    			
    		    //createFeatures(inst);
    			lt.add(new DependencyInstance(inst));		    
    			
    			inst = reader.nextInstance();
    			cnt++;
    			//if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;
    			//if (cnt % 1000 == 0)
    			//	System.out.printf("%d ", cnt);
    		}
    				
    		reader.close();
    		//if (options.maxNumSent != -1 && cnt >= options.maxNumSent) break;
        }
        System.out.println("Done.");
				
		DependencyInstance[] insts = shuffle(lt);
		
		System.out.printf("%d [%d ms]%n", insts.length, System.currentTimeMillis() - start);
	    
		return insts;
	}
	
    public DependencyInstance[] shuffle(ArrayList<DependencyInstance> lt) {
    	int n = options.maxNumSent == -1 ? lt.size() : Math.min(options.maxNumSent, lt.size());
    	boolean[] used = new boolean[n];
    	DependencyInstance[] ret = new DependencyInstance[n];
    	int id = 0;
    	Random r = new Random(0);
    	for (int i = 0; i < n; ++i) {
    		id = (id + r.nextInt(n)) % n;
    		while (used[id]) {
    			id = (id + 1) % n;
    		}
    		//id = i;
    		used[id] = true;
    		ret[i] = lt.get(id);
    	}
    	return ret;
    }

    public DependencyInstance createInstance(DependencyReader reader) throws IOException 
    {
    	DependencyInstance inst = reader.nextInstance();
    	if (inst == null) return null;
    	
    	inst.setInstIds(dictionaries);
	    
	    return inst;
    }
}
