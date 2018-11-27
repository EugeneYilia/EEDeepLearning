package com.jstarcraft.module.math.algorithm.distribution;

import org.apache.commons.math3.distribution.AbstractRealDistribution;

public class ContinuousProbability implements Probability<Double> {

	private final AbstractRealDistribution distribution;

	public ContinuousProbability(AbstractRealDistribution distribution) {
		this.distribution = distribution;
	}

	/**
	 * Returns the probability density function (PDF) of this distribution
	 * evaluated at the specified point {@code x}. In general, the PDF is the
	 * derivative of the {@link #cumulativeDistribution(double) CDF}. If the
	 * derivative does not exist at {@code x}, then an appropriate replacement
	 * should be returned, e.g. {@code Double.POSITIVE_INFINITY},
	 * {@code Double.NaN}, or the limit inferior or limit superior of the
	 * difference quotient.
	 *
	 * @param value
	 *            the point at which the PDF is evaluated
	 * @return the value of the probability density function at point {@code x}
	 */
	// TODO 概率密度函数(probability density function)是连续型随机变量在各个特定值的概率.
	public double density(Double value) {
		return distribution.density(value);
	}

	@Override
	public double cumulativeDistribution(Double value) {
		return distribution.cumulativeProbability(value);
	}

	@Override
	public double cumulativeDistribution(Double minimum, Double maximum) {
		return distribution.probability(minimum, maximum);
	}

	@Override
	public Double inverseDistribution(double probability) {
		return distribution.inverseCumulativeProbability(probability);
	}

	@Override
	public Double sample() {
		return distribution.sample();
	}

	@Override
	public Double getMaximum() {
		return distribution.getSupportUpperBound();
	}

	@Override
	public Double getMinimum() {
		return distribution.getSupportLowerBound();
	}

	@Override
	public double getMean() {
		return distribution.getNumericalMean();
	}

	@Override
	public double getVariance() {
		return distribution.getNumericalVariance();
	}

	@Override
	public void setSeed(long seed) {
		distribution.reseedRandomGenerator(seed);
	}

}
