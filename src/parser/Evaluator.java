package parser;

import java.io.IOException;
import java.util.regex.Pattern;

public class Evaluator {
	int uas, las, tot;
	int whole, nsents;
	
	boolean learnLabel;
	String[] labels;
	
	// Note: this requires jdk version >= 1.7
	public static Pattern puncRegex = Pattern.compile("[\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS);
	
	// Note: use this regex if jdk version < 1.7
	//public static Pattern puncRegex = Pattern.compile("[-!\"%&'()*,./:;?@\\[\\]_{}、，。！]+");
	
	
	public Evaluator(Options options, DependencyPipe pipe)
	{
		uas = las = tot = 0;
		whole = nsents = 0;
		learnLabel = options.learnLabel;
		if (learnLabel) labels = pipe.types;
	}
	
	
	public double UAS()
	{
		return uas/(tot+1e-20);
	}
	
	public double LAS()
	{
		return las/(tot+1e-20);
	}
	
	public double CAS()
	{
		return whole/(nsents+1e-20);
	}
	
	
	public void add(DependencyInstance gold, DependencyInstance predict, boolean evalWithPunc)
	{
		evaluateDependencies(gold, predict, evalWithPunc);
	}
	
    public void evaluateDependencies(DependencyInstance gold, 
    		DependencyInstance pred, boolean evalWithPunc) 
    {
    	++nsents;
    	int tt = 0, ua = 0, la = 0;
    	for (int i = 1, N = gold.length; i < N; ++i) {

            if (!evalWithPunc) {
            
            	//if (puncRegex.matcher(gold.forms[i]).matches()) continue;
            	//if (gold.forms[i].matches("[-!\"%&'()*,./:;?@\\[\\]_{}、]+")) continue;
            	if (gold.postags[i].equals("."))
            		continue;
            }
            ++tt;
    		if (gold.heads[i] == pred.heads[i]) {
    			++ua;
    			if (learnLabel && gold.deplbs[i].equals(labels[pred.deplbids[i]])) ++la;
    		}
    	
    		//System.out.println(gold.forms[i] + "\t" + gold.postags[i] + "\t" + pred.heads[i] + "\t" + pred.deplbids[i]);
    	}    	
    	
    	tot += tt;
    	uas += ua;
    	las += la;
    	whole += (tt == ua) && (tt == la || !learnLabel) ? 1 : 0;
    }

}
