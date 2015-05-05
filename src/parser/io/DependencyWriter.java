package parser.io;

import java.io.*;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;
import parser.Options.Dataset;

public abstract class DependencyWriter {
	BufferedWriter writer;
	Options options;
	String[] labels;
	boolean first, isLabeled;
	int lang;
	
	public static DependencyWriter createDependencyWriter(Options options, int lang, DependencyPipe pipe) {
		Dataset dataset = options.dataset;
		if (dataset == Dataset.CoNLL_UNI) {
			return new ConllUniWriter(options, lang, pipe);
		} else {
			System.out.printf("!!!!! Unsupported file format: %s%n", dataset.name());
			return new ConllUniWriter(options, lang, pipe);
		}
	}
	
	public abstract void writeInstance(DependencyInstance inst) throws IOException;
	
	public void startWriting(String file) throws IOException {
		writer = new BufferedWriter(new FileWriter(file));
		first = true;
		isLabeled = options.learnLabel;
	}
	
	public void close() throws IOException {
		if (writer != null) writer.close();
	}

}
