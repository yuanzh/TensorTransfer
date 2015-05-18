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
	    
	    HEAD_EMB,
	    MOD_EMB,

	    // Bare
	    DIST,
	    B_HP,
	    B_MP,
	    B_HP_MP,
	    B_HEAD_EMB,
	    B_MOD_EMB,
	    
	    // selectively shared
	    SV_NOUN,
	    SV_PRON,
	    VO_NOUN,
	    VO_PRON,
	    ADP_NOUN,
	    ADP_PRON,
	    GEN,
	    ADJ,
	    
	    HW_MP,
	    HW_HP_MP,
	    MW_HP,
	    MW_HP_MP,

	    /*************************************************
		 * Supervised Model Feature 
		 * ***********************************************/
		
	    // posL posIn posR
	    L_HP_BP_MP,			//CORE_POS_PC,	    
	    					//CORE_POS_XPC,
	    
	    // posL-1 posL posR posR+1
	    L_HPp_HP_MP_MPn,		//CORE_POS_PT0,
	    L_HP_MP_MPn,			//CORE_POS_PT1,
	    L_HPp_HP_MP,			//CORE_POS_PT2,
	    L_HPp_MP_MPn,			//CORE_POS_PT3,
	    L_HPp_HP_MPn,			//CORE_POS_PT4,
    
	    // posL posL+1 posR-1 posR
	    L_HP_HPn_MPp_MP,		//CORE_POS_APT0,
	    L_HP_MPp_MP,			//CORE_POS_APT1,
	    L_HP_HPn_MP,			//CORE_POS_APT2,
	    L_HPn_MPp_MP,			//CORE_POS_APT3,
	    L_HP_HPn_MPp,			//CORE_POS_APT4,
	    
	    // posL-1 posL posR-1 posR
	    // posL posL+1 posR posR+1
	    L_HPp_HP_MPp_MP,		//CORE_POS_BPT,
	    L_HP_HPn_MP_MPn,		//CORE_POS_CPT,

	    // unigram (form, lemma, pos, coarse_pos, morphology) 
	    L_CORE_HEAD_WORD,
	    L_CORE_HEAD_POS,
	    L_CORE_MOD_WORD,
	    L_CORE_MOD_POS,
	    
	    // bigram  [word|lemma]-cross-[pos|cpos|mophlogy](-cross-distance)
	    L_HW_MW_HP_MP,			//CORE_BIGRAM_A,
	    L_MW_HP_MP,				//CORE_BIGRAM_B,
	    L_HW_HP_MP,				//CORE_BIGRAM_C,
	    L_MW_HP,					//CORE_BIGRAM_D,
	    L_HW_MP,					//CORE_BIGRAM_E,
	    L_HW_HP,					//CORE_BIGRAM_H,
	    L_MW_MP,					//CORE_BIGRAM_K,
	    L_HW_MW,					//CORE_BIGRAM_F,
	    L_HP_MP,					//CORE_BIGRAM_G,
	    
	    FEATURE_TEMPLATE_END;
		
		public final static int numArcFeatBits = Utils.log2(FEATURE_TEMPLATE_END.ordinal());
	}
}
