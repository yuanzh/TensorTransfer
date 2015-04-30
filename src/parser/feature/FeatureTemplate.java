package parser.feature;

import utils.Utils;

public class FeatureTemplate {
	
	/**
	 * "H"	: head
	 * "M"	: modifier
	 * "B"	: in-between tokens
	 * 
	 * "P"	: pos tag
	 * "W"	: word form or lemma
	 * "EMB": word embedding (word vector)
	 * 
	 * "p": previous token
	 * "n": next token
	 *
	 */
	
	public enum Arc {
		
		FEATURE_TEMPLATE_START,
	    
		/*************************************************
		 * Arc feature inspired by MST parser 
		 * ***********************************************/
		
		// Delexicalized MSTParser
		ATTDIST,
		
		HP,
		MP,
		
		HP_MP,
	    HPp_HP_MP,
	    HP_HPn_MP,
	    HP_MPp_MP,	
	    HP_MP_MPn,			
	    HPp_HP_MP_MPn,		
	    HP_HPn_MP_MPn,		
	    HP_HPn_MPp_MP,		
	    HPp_HP_MPp_MP,		
		
	    HP_BP_MP,			
	    
	    // Bare
	    DIST,
	    B_HP,
	    B_MP,
	    B_HP_MP,
	    
	    // selectively shared
	    SV_NOUN,
	    SV_PRON,
	    VO_NOUN,
	    VO_PRON,
	    ADP_NOUN,
	    ADP_PRON,
	    GEN,
	    ADJ,
	    
	    FEATURE_TEMPLATE_END;
		
		public final static int numArcFeatBits = Utils.log2(FEATURE_TEMPLATE_END.ordinal());
	}
}
