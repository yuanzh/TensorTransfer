package parser.io;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import parser.DependencyInstance;
import parser.Options;
import parser.Options.Dataset;

public abstract class DependencyReader {
	BufferedReader reader;
	Options options;
	int lang;
	
	public static DependencyReader createDependencyReader(Options options, int lang) {
		Dataset dataset = options.dataset;
		if (dataset == Dataset.CoNLL_UNI) {
			return new ConllUniReader(options, lang);
		} else {
			System.out.printf("!!!!! Unsupported file format: %s%n", dataset.name());
			return new ConllUniReader(options, lang);
		}
	}
	
	public abstract DependencyInstance nextInstance() throws IOException;
	
	public void startReading(String file) throws IOException {
		reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF8"));
	}
	
	public void close() throws IOException { if (reader != null) reader.close(); }
}
