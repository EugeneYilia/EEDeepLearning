package com.jstarcraft.module.math.algorithm;

/**
 * Gaussian
 * 
 * probabilityDensityFunction
 * 
 * cumulativeDistributionFunction
 *
 * @author WangYuFeng
 */
// TODO 将保留probabilityDensity方法并迁移到MathUtility.
@Deprecated
public class Gaussian {

	// return pdf(x) = standard Gaussian pdf
	public static float probabilityDensity(float value) {
		return (float) (Math.exp(-value * value / 2F) / Math.sqrt(2F * Math.PI));
	}

	// return pdf(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
	public static float probabilityDensity(float value, float mean, float standardDeviation) {
		return probabilityDensity((value - mean) / standardDeviation) / standardDeviation;
	}

	// return cdf(z) = standard Gaussian cdf using Taylor approximation
	public static float cumulativeDistribution(float value) {
		if (value < -8F) {
			return 0F;
		}
		if (value > 8F) {
			return 1F;
		}
		float sum = 0F, term = value;
		for (int index = 3; sum + term != sum; index += 2) {
			sum = sum + term;
			term = term * value * value / index;
		}
		return 0.5F + sum * probabilityDensity(value);
	}

	// return cdf(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
	public static float cumulativeDistribution(float value, float mean, float standardDeviation) {
		return cumulativeDistribution((value - mean) / standardDeviation);
	}

	/**
	 * Compute z for standard normal such that cdf(z) = y via bisection search
	 *
	 * @param value
	 *            parameter of the function
	 * @return z for standard normal such that cdf(z) = y
	 */
	public static float inverseCDF(float value) {
		return inverseCDF(value, MathUtility.EPSILON, -8F, 8F);
	}

	private static float inverseCDF(float value, float delta, float minimum, float maximum) {
		float median = minimum + (maximum - minimum) / 2F;
		if (maximum - minimum < delta) {
			return median;
		}
		if (cumulativeDistribution(median) > value) {
			return inverseCDF(value, delta, minimum, median);
		} else {
			return inverseCDF(value, delta, median, maximum);
		}
	}

	/**
	 * Compute z for standard normal such that cdf(z, mu, sigma) = y via bisection
	 * search
	 *
	 * @param value
	 *            the given value for function
	 * @param mean
	 *            mu parameter
	 * @param standardDeviation
	 *            sigma parameter
	 * @return z for standard normal such that cdf(z, mu, sigma) = y
	 */
	public static float inverseCDF(float value, float mean, float standardDeviation) {
		return inverseCDF(value, mean, standardDeviation, MathUtility.EPSILON, (mean - 8F * standardDeviation), (mean + 8F * standardDeviation));
	}

	private static float inverseCDF(float value, float mean, float standardDeviation, float delta, float minimum, float maximum) {
		float median = minimum + (maximum - minimum) / 2F;
		if (maximum - minimum < delta) {
			return median;
		}
		if (cumulativeDistribution(median, mean, standardDeviation) > value) {
			return inverseCDF(value, mean, standardDeviation, delta, minimum, median);
		} else {
			return inverseCDF(value, mean, standardDeviation, delta, median, maximum);
		}
	}

}
