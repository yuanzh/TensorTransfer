package parser.decoding;

import parser.DependencyInstance;
import parser.FeatureData;
import parser.Options;
import parser.Options.FeatureMode;
import utils.Utils;

public abstract class DependencyDecoder {
	
	Options options;
	
	public static DependencyDecoder createDependencyDecoder(Options options)
	{
		if (options.featureMode != FeatureMode.Basic && options.projective) {
			System.out.println("WARNING: high-order projective parsing not supported. "
					+ "Switched to non-projective parsing.");
			options.projective = false;
		}
		
		if (options.featureMode == FeatureMode.Basic) {
			if (!options.projective)
				return new ChuLiuEdmondDecoder(options);
			else
				return new CKYDecoder(options);			
		} else
			Utils.ThrowException("not implemented yet!");
		
		return null;
	}
    
    public void shutdown()
    {
    }

	public abstract DependencyInstance decode(DependencyInstance inst, FeatureData fd);

}
