package parser.io;

import java.io.IOException;
import java.util.ArrayList;

import parser.DependencyInstance;
import parser.Options;

public class ConllUniReader extends DependencyReader {

	public ConllUniReader(Options options, int lang) {
		this.options = options;
		this.lang = lang;
	}

	@Override
	public DependencyInstance nextInstance() throws IOException {
		
	    ArrayList<String[]> lstLines = new ArrayList<String[]>();

	    String line = reader.readLine();
	    while (line != null && !line.equals("") && !line.startsWith("*")) {
	    	if (!line.startsWith("#")) {
	    		String[] data = line.split("\t");
	    		if (!data[0].contains("-")) {
	    			lstLines.add(data);
	    		}
	    	}
	    	line = reader.readLine();
	    }
	    
	    if (lstLines.size() == 0) return null;
	    
	    int length = lstLines.size();
	    String[] forms = new String[length + 1];
	    String[] pos = new String[length + 1];
	    String[] deplbs = new String[length + 1];
	    int[] heads = new int[length + 1];
	    
	    forms[0] = "<root>";
	    pos[0] = "<root-POS>";
	    deplbs[0] = "<no-type>";
	    heads[0] = -1;
	    
	    // 3 eles ele pron pron-pers M|3P|NOM 4 SUBJ _ _
	    // ID FORM LEMMA COURSE-POS FINE-POS FEATURES HEAD DEPREL PHEAD PDEPREL
	    for (int i = 1; i < length + 1; ++i) {
	    	String[] parts = lstLines.get(i-1);
	    	forms[i] = parts[1];
	    	pos[i] = parts[3];
	    	heads[i] = Integer.parseInt(parts[6]);
	    	deplbs[i] = parts[7];
	    }
	    
		return new DependencyInstance(lang, forms, pos, heads, deplbs);
	}

}
