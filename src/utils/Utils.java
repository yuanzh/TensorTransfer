package utils;

import java.util.Random;

public final class Utils {
	
	public static Random rnd = new Random(0);
	
	public static void Assert(boolean assertion) 
	{
		if (!assertion) {
			(new Exception()).printStackTrace();
			System.exit(1);
		}
	}
		
	public static void ThrowException(String msg) {
		throw new RuntimeException(msg);
	}
		
	public static int log2(long x) 
	{
		long y = 1;
		int i = 0;
		while (y < x) {
			y = y << 1;
			++i;
		}
		return i;
	}
	
	public static double logSumExp(double x, double y) 
	{
		if (x == Double.NEGATIVE_INFINITY && x == y)
			return Double.NEGATIVE_INFINITY;
		else if (x < y)
			return y + Math.log1p(Math.exp(x-y));
		else 
			return x + Math.log1p(Math.exp(y-x));
	}
	
	public static double[] getRandomUnitVector(int length) 
	{
		double[] vec = new double[length];
		double sum = 0;
		for (int i = 0; i < length; ++i) {
			vec[i] = rnd.nextDouble() - 0.5;
			sum += vec[i] * vec[i];
		}
		double invSqrt = 1.0 / Math.sqrt(sum);
		for (int i = 0; i < length; ++i) 
			vec[i] *= invSqrt;
		return vec;
	}
	
	public static double[] getRandomVector(int length, double range) 
	{
		double[] vec = new double[length];
		for (int i = 0; i < length; ++i) {
			vec[i] = rnd.nextDouble() * range * 2 - range;
			//vec[i] = range;
		}
		return vec;
	}
	
	public static double squaredSum(double[] vec) 
	{
		double sum = 0;
		for (int i = 0, N = vec.length; i < N; ++i)
			sum += vec[i] * vec[i];
		return sum;
	}
	
	public static void normalize(double[] vec) 
	{
		double coeff = 1.0 / Math.sqrt(squaredSum(vec));
		for (int i = 0, N = vec.length; i < N; ++i)
			vec[i] *= coeff;
	}
	
	public static double max(double[] vec) 
	{
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			max = Math.max(max, vec[i]);
		return max;
	}
	
	public static double min(double[] vec) 
	{
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			min = Math.min(min, vec[i]);
		return min;
	}
	
	public static double[] dot(double[]... vecs)
	{
		Utils.Assert(vecs.length > 0);
		int N = vecs[0].length;
		double[] ret = new double[N];
		for (int i = 0; i < N; ++i) {
			ret[i] = 1.0;
			for (double[] vec : vecs)
				ret[i] *= vec[i];
		}
		return ret;
	}
	
	public static double[] dot_s(double[] ret, double[]... vecs) {
		int N = ret.length;
		for (int i = 0; i < N; ++i) {
			double r = 1.0;
			for (double[] vec : vecs)
				r *= vec[i];
			ret[i] = r;
		}
		return ret;
	}
	
	public static double sum(double[] vec) {
		double sum = 0.0;
		for (int i = 0, L = vec.length; i < L; ++i)
			sum += vec[i];
		return sum;
	}
	
	public static void block() {
		try { System.in.read(); } catch (Exception e) { e.printStackTrace(); }
	}
}
