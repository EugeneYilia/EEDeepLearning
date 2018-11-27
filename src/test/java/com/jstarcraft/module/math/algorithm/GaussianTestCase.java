package com.jstarcraft.module.math.algorithm;

import org.junit.Assert;
import org.junit.Test;

public class GaussianTestCase {

	@Test
	public void testCDF() {
		// This prints out the values of the probability density function for
		// N(2.0.0.6)
		// A graph of this is here:
		// http://www.cs.bu.edu/fac/snyder/cs237/Lecture%20Materials/GaussianExampleJava.png
		float mean = 2F;
		float standardDeviation = 1.5F;
		System.out.println("PDF for N(2.0, 1.5) in range [-4..8]:");
		for (float value = -4F; value <= 8F; value += 0.2F) {
			System.out.format("%.1f\t%.4f\n", value, Gaussian.probabilityDensity(value, mean, standardDeviation));
		}

		// This prints out the values of the cumulative density function for
		// N(2.0.0.6)
		// A graph of this is here:
		// http://www.cs.bu.edu/fac/snyder/cs237/Lecture%20Materials/GaussianExample2Java.png
		System.out.println("CDF for N(2.0, 1.5) in range [-4..8]:");
		for (float value = -4F; value <= 8F; value += 0.2F) {
			System.out.format("%.1f\t%.4f\n", value, Gaussian.cumulativeDistribution(value, mean, standardDeviation));
		}

		// Calculates the probability that for N(2.0, 0.6), the random variable
		// produces a value less than 3.45
		System.out.format("\nIf X ~ N(2.0, 1.5), then P(X <= 3.2) is %.4f\n", Gaussian.cumulativeDistribution(3.2F, mean, standardDeviation));

		// Calculates the value x for X ~ N(2.0, 0.6) which is the 78.81% cutoff
		// (i.e., 78.81% of the values lie below x and 21.19% above).
		System.out.format("\nIf X ~ N(2.0, 1.5), then x such that P(X <= x ) = 0.7881 is %.4f\n", Gaussian.inverseCDF(0.7881F, mean, standardDeviation));
		float value = 3.2F;
		value = Gaussian.cumulativeDistribution(value, mean, standardDeviation);
		value = Gaussian.inverseCDF(value, mean, standardDeviation);
		Assert.assertTrue(MathUtility.equal(value, 3.2F));
	}

}
