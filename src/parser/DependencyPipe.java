package parser;

import static utils.DictionarySet.DictionaryTypes.DEPLABEL;
import static utils.DictionarySet.DictionaryTypes.POS;
import static utils.DictionarySet.DictionaryTypes.WORD;

import java.io.*;
import java.util.*;

import parser.Options.Dataset;
import parser.feature.FeatureFactory;
import parser.feature.FeatureRepo;
import parser.io.DependencyReader;
import parser.io.DependencyWriter;
import parser.tensor.ParameterNode;

import utils.Dictionary;
import utils.DictionarySet;
import utils.Distribution;
import utils.TypologicalInfo;
import utils.Utils;
import utils.WordVector;

public class DependencyPipe implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

    public Options options;
    public DictionarySet dictionaries;
    public FeatureFactory ff;
    public transient TypologicalInfo typo;
    public transient WordVector wv;
    public transient FeatureRepo fr;
        
    public String[] types;					// array that maps label index to label string
    public String[] poses;

    public DependencyPipe(Options options) throws IOException 
	{
		dictionaries = new DictionarySet();
		ff = new FeatureFactory(options);
		typo = new TypologicalInfo(options);
		if (options.lexical)
			wv = new WordVector(options);
		
		ff.typo = typo;
		ff.wv = wv;
		
		dictionaries.wv = wv;
		
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
        	//if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        	//	continue;
        	
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
		Utils.Assert(ff.TOKEN_START == dictionaries.lookupIndex(WORD, "#TOKEN_START#") - 1);
        Utils.Assert(ff.TOKEN_END == dictionaries.lookupIndex(WORD, "#TOKEN_END#") - 1); 
        Utils.Assert(ff.TOKEN_MID == dictionaries.lookupIndex(WORD, "#TOKEN_MID#") - 1);
		
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
		dictionaries.stopGrowth(WORD);
				
		ff.tagNumBits = Math.max(Utils.log2(dictionaries.size(POS) + 1), typo.bit + 1);
		ff.depNumBits = Utils.log2(dictionaries.size(DEPLABEL)*2 + 1);
		ff.wordNumBits = Utils.log2(dictionaries.size(WORD)*2 + 1);
		
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
			//System.out.println(id + ": " + types[id - 1]);
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
		System.out.printf("Lexical items: %d (%d bits)%n", 
				dictionaries.size(WORD), ff.wordNumBits);
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
        	//if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        	//	continue;
        	
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
		System.out.printf("Num of Syntactic Features: %d %d%n", ff.featureHashSet.size(), ff.featureIDSet.size());
	}

    public void closeAlphabets() 
    {
		ff.closeAlphabets();
    }

    public void outputTrainingData(DependencyInstance[] insts) throws IOException {
    	DependencyWriter writer = DependencyWriter.createDependencyWriter(options, options.targetLang, this);
		writer.startWriting(constructTrainFileName(options.targetLang) + ".rand50");
		for (int i = 0; i < insts.length; ++i)
			writer.writeInstance(insts[i]);
		writer.close();
		
		System.exit(0);
    }
    
    public DependencyInstance[] createInstances() throws IOException 
    {
    	
    	long start = System.currentTimeMillis();
    	System.out.print("Creating instances ... ");
    	
		ArrayList<DependencyInstance> lt = new ArrayList<DependencyInstance>();
		int cnt = 0;
        for (int l = 0; l < options.langString.length; ++l) {
        	//if (l == options.targetLang)
        	//if (!options.langString[l].equals("es"))
        	//	continue;
        	
        	String file = constructTrainFileName(l);
        	System.out.print(" " + options.langString[l] + " ");
            
        	DependencyReader reader = DependencyReader.createDependencyReader(options, l);
    		reader.startReading(file);

    		DependencyInstance inst = reader.nextInstance();
    						
    		while(inst != null) {
    			
    			inst.setInstIds(dictionaries);
    			
    		    //createFeatures(inst);
    			lt.add(new DependencyInstance(inst));
    			//lt.add(inst);
    			
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
		
        //Distribution dist = new Distribution(lt, this, options);
		//DependencyInstance[] insts = shuffle(lt, dist);
        //DependencyInstance[] insts = greedyShuffle(lt, dist);
        //DependencyInstance[] tmpinsts = lengthShuffle(lt, dist);
        DependencyInstance[] tmpinsts = greedyShuffle2(lt);
        
        DependencyInstance[] insts = dupAndRandom(tmpinsts);
        //DependencyInstance[] insts = removeUnsup(tmpinsts);
        //DependencyInstance[] insts = new DependencyInstance[options.supSent];
        //for (int i = 0; i < options.supSent; ++i)
        //	insts[i] = tmpinsts[i];
		
		System.out.printf("%d [%d ms]%n", insts.length, System.currentTimeMillis() - start);
	    
		//outputTrainingData(insts);
		
		return insts;
	}
    
    public DependencyInstance[] removeUnsup(DependencyInstance[] lt) {
    	int size = options.supSent;
    	DependencyInstance[] tmp = new DependencyInstance[size];
    	for (int i = 0; i < options.supSent; ++i) {
    		tmp[i] = lt[i];
    	}
    	return tmp;
    }
    
    public DependencyInstance[] dupAndRandom(DependencyInstance[] lt) {
    	int copy = (int)((lt.length + 0.0) / options.supSent * 0.01);
    	int size = lt.length + (copy - 1) * options.supSent;
    	DependencyInstance[] tmp = new DependencyInstance[size];
    	for (int i = 0; i < options.supSent; ++i) {
    		for (int j = 0; j < copy; ++j) {
    			tmp[i * copy + j] = lt[i];
    		}
    	}
    	for (int i = options.supSent; i < lt.length; ++i)
    		tmp[options.supSent * (copy - 1) + i] = lt[i];
    	
    	Random r = new Random(0);
    	boolean[] used = new boolean[size];
    	DependencyInstance[] ret = new DependencyInstance[size];
    	int id = 0;
    	for (int i = 0; i < size; ++i) {
    		id = (id + r.nextInt(size)) % size;
    		while (used[id]) {
    			id = (id + 1) % size;
    		}
    		used[id] = true;
    		ret[i] = tmp[id];
    	}
    	return ret;
    }
    
    public DependencyInstance[] greedyShuffle(ArrayList<DependencyInstance> lt, Distribution dist) {
    	int size = lt.size();
    	int supNum = 0;
    	for (int i = 0; i < size; ++i)
    		if (lt.get(i).lang == options.targetLang)
    			supNum++;
    	if (options.supSent == -1)
    		options.supSent = supNum;
    	
    	int n = options.maxNumSent == -1 ? lt.size() - supNum + options.supSent : Math.min(options.maxNumSent, lt.size() - supNum + options.supSent);
    	boolean[] used = new boolean[size];
    	DependencyInstance[] ret = new DependencyInstance[n];
    	
    	// get supervised data
    	double[] supScore = new double[options.supSent];
    	Arrays.fill(supScore, Double.NEGATIVE_INFINITY);
    	for (int i = 0; i < size; ++i) {
    		if (lt.get(i).lang != options.targetLang)
    			continue;
    		double score = dist.getScore(lt.get(i));
    		int j = 0;
    		for (; j < options.supSent; ++j)
    			if (score > supScore[j] + 1e-8)
    				break;
    		if (j < options.supSent) {
    			for (int k = options.supSent - 1; k > j; --k) {
    				supScore[k] = supScore[k - 1];
    				ret[k] = ret[k - 1];
    			}
    			supScore[j] = score;
    			ret[j] = lt.get(i);
    			used[i] = true;
    		}
    	}
    	System.out.println("supervised data: " + options.supSent);
    	for (int i = 0; i < options.supSent; ++i)
    		System.out.print("  " + supScore[i]);
    	System.out.println();
    	
    	//get unsupervised data
    	Random r = new Random(0);
    	int id = 0;
    	for (int i = options.supSent; i < n; ++i) {
    		id = (id + r.nextInt(size)) % size;
    		while (used[id] || lt.get(id).lang == options.targetLang) {
    			id = (id + 1) % size;
    		}
    		used[id] = true;
    		ret[i] = lt.get(id);
    	}
    
    	return ret;
    }
	
    public DependencyInstance[] greedyShuffle2(ArrayList<DependencyInstance> lt) {
    	ArrayList<DependencyInstance> lt2 = new ArrayList<DependencyInstance>();
    	int size = lt.size();
    	int supNum = 0;
    	for (int i = 0; i < size; ++i)
    		if (lt.get(i).lang == options.targetLang) {
    			supNum++;
    			lt2.add(lt.get(i));
    		}
    	if (options.supSent == -1)
    		options.supSent = supNum;
    	
    	int n = options.maxNumSent == -1 ? lt.size() - supNum + options.supSent : Math.min(options.maxNumSent, lt.size() - supNum + options.supSent);
    	boolean[] used = new boolean[size];
    	DependencyInstance[] ret = new DependencyInstance[n];
    	
    	//get unsupervised data
    	Random r = new Random(0);
    	int id = 0;
    	for (int i = options.supSent; i < n; ++i) {
    		id = (id + r.nextInt(size)) % size;
    		while (used[id] || lt.get(id).lang == options.targetLang) {
    			id = (id + 1) % size;
    		}
    		used[id] = true;
    		ret[i] = lt.get(id);
    		lt2.add(lt.get(id));
    	}
    	
    	Distribution dist = new Distribution(lt2, this, options);
    
    	// get supervised data
    	double[] supScore = new double[options.supSent];
    	for (int i = 0; i < options.supSent; ++i) {
    		double maxSupScore = Double.NEGATIVE_INFINITY;
    		int maxID = -1;
    		for (int j = 0; j < size; ++j) {
        		if (lt.get(j).lang != options.targetLang || used[j])
        			continue;
        		double score = dist.getScore(lt.get(j));
        		if (score > maxSupScore + 1e-6) {
        			maxSupScore = score;
        			ret[i] = lt.get(j);
        			maxID = j;
        		}
    		}
    		used[maxID] = true;
    		supScore[i] = maxSupScore;
    		dist.addCount(ret[i], (n + 0.0) / options.supSent * 0.1, true);
    	}

    	System.out.println("supervised data: " + options.supSent);
    	for (int i = 0; i < options.supSent; ++i)
    		System.out.print("  " + supScore[i]);
    	System.out.println();
    	
    	return ret;
    }
	
    public DependencyInstance[] lengthShuffle(ArrayList<DependencyInstance> lt, Distribution dist) {
    	int size = lt.size();
    	int supNum = 0;
    	for (int i = 0; i < size; ++i)
    		if (lt.get(i).lang == options.targetLang)
    			supNum++;
    	if (options.supSent == -1)
    		options.supSent = supNum;
    	
    	int n = options.maxNumSent == -1 ? lt.size() - supNum + options.supSent : Math.min(options.maxNumSent, lt.size() - supNum + options.supSent);
    	boolean[] used = new boolean[size];
    	DependencyInstance[] ret = new DependencyInstance[n];
    	
    	// get supervised data
    	int[] len = new int[options.supSent];
    	Arrays.fill(len, 0);
    	for (int i = 0; i < size; ++i) {
    		if (lt.get(i).lang != options.targetLang)
    			continue;
    		int length = lt.get(i).length;
    		int j = 0;
    		for (; j < options.supSent; ++j)
    			if (length > len[j])
    				break;
    		if (j < options.supSent) {
    			for (int k = options.supSent - 1; k > j; --k) {
    				len[k] = len[k - 1];
    				ret[k] = ret[k - 1];
    			}
    			len[j] = length;
    			ret[j] = lt.get(i);
    			used[i] = true;
    		}
    	}
    	System.out.println("supervised data: " + options.supSent);
    	for (int i = 0; i < options.supSent; ++i)
    		System.out.print("  " + dist.getScore(ret[i]));
    	System.out.println();
    	
    	//get unsupervised data
    	Random r = new Random(0);
    	int id = 0;
    	for (int i = options.supSent; i < n; ++i) {
    		id = (id + r.nextInt(size)) % size;
    		while (used[id] || lt.get(id).lang == options.targetLang) {
    			id = (id + 1) % size;
    		}
    		used[id] = true;
    		ret[i] = lt.get(id);
    	}
    
    	return ret;
    }
	
    public DependencyInstance[] shuffle(ArrayList<DependencyInstance> lt, Distribution dist) {
    	int size = lt.size();
    	int supNum = 0;
    	for (int i = 0; i < size; ++i)
    		if (lt.get(i).lang == options.targetLang)
    			supNum++;
    	if (options.supSent == -1)
    		options.supSent = supNum;
    	
    	int n = options.maxNumSent == -1 ? lt.size() - supNum + options.supSent : Math.min(options.maxNumSent, lt.size() - supNum + options.supSent);

    	boolean[] used = new boolean[size];
    	DependencyInstance[] ret = new DependencyInstance[n];
    	int id = 0;
    	Random r = new Random(0);
    	int supCnt = 0;
    	double[] score = new double[options.supSent];
    	for (int i = 0; i < options.supSent; ++i) {
    		id = (id + r.nextInt(size)) % size;
    		while (used[id] || lt.get(id).lang != options.targetLang) {
    			id = (id + 1) % size;
    		}
   			score[supCnt] = dist.getScore(lt.get(id));
   			supCnt++;
    		//id = i;
    		used[id] = true;
    		ret[i] = lt.get(id);
    	}
    	for (int i = options.supSent; i < n; ++i) {
    		//System.out.print(" " + i + " ");
    		id = (id + r.nextInt(size)) % size;
    		while (used[id] || lt.get(id).lang == options.targetLang) {
    			//System.out.print(lt.get(id).lang + "/" + id);
    			id = (id + 1) % size;
    		}
    		used[id] = true;
    		ret[i] = lt.get(id);
    	}
    	System.out.println("supervised data: " + supCnt);
    	for (int i = 0; i < options.supSent; ++i)
    		System.out.print("  " + score[i]);
    	System.out.println();
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
