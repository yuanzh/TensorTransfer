package parser;

import static utils.DictionarySet.DictionaryTypes.DEPLABEL;
import static utils.DictionarySet.DictionaryTypes.POS;
import static utils.DictionarySet.DictionaryTypes.WORDVEC;

import java.io.IOException;
import java.io.Serializable;
import java.util.regex.Pattern;
import utils.DictionarySet;

public class DependencyInstance implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	public static Pattern puncRegex = Pattern.compile("[\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS);
	public static Pattern numberRegex = Pattern.compile("[\\p{Punct}0-9]+", Pattern.UNICODE_CHARACTER_CLASS);

	public int length;
	
	// language id
	public int lang;

	// FORM: the forms - usually words, like "thought", mainly used for wordvec
	public String[] forms;

	// COARSE-POS: the universal part-of-speech tags
	public String[] postags;

	// HEAD: the IDs of the heads for each element
	public int[] heads;

	// DEPREL: the dependency labels, e.g. "SUBJ"
	public String[] deplbs;
	
	public int[] postagids;
	public int[] deplbids;
	public int[] wordVecIds;

    public DependencyInstance(int lang, String[] forms, String[] postags, int[] heads) {
    	this.length = forms.length;
    	this.lang = lang;
    	this.forms = forms;
    	this.heads = heads;
	    this.postags = postags;
    }
    
    public DependencyInstance(int lang, String[] forms, String[] postags, int[] heads, String[] deplbs) {
    	this(lang, forms, postags, heads);
    	this.deplbs = deplbs;    	
    }
    
    public DependencyInstance(DependencyInstance a) {
    	//this(a.forms, a.lemmas, a.cpostags, a.postags, a.feats, a.heads, a.deprels);
    	length = a.length;
    	lang = a.lang;
    	heads = a.heads;
    	postagids = a.postagids;
    	deplbids = a.deplbids;
    	wordVecIds = a.wordVecIds;
    }
    
    
    public void setInstIds(DictionarySet dicts) {
    	    	
		postagids = new int[length];
		deplbids = new int[length];
		
    	for (int i = 0; i < length; ++i) {
			postagids[i] = dicts.lookupIndex(POS, postags[i]) - 1;		// zero-based
			deplbids[i] = dicts.lookupIndex(DEPLABEL, deplbs[i]) - 1;	// zero-based
			//System.out.println(deplbids[i] + "\t" + deplbs[i]);
    	}
    	//try { System.in.read(); } catch (IOException e) { e.printStackTrace(); }
    	
		if (dicts.size(WORDVEC) > 0) {
			//TODO: add wordvec
		}
    }

    private String normalize(String s) {
    	if (numberRegex.matcher(s).matches()) {
    		return "<num>";
    	}
		return s;
    }
}
