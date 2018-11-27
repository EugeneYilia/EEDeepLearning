package com.jstarcraft.module.math.algorithm.distribution;

import org.apache.commons.math3.distribution.AbstractIntegerDistribution;

public class DiscreteProbability implements Probability<Integer> {

	private final AbstractIntegerDistribution distribution;

	public DiscreteProbability(AbstractIntegerDistribution distribution) {
		this.distribution = distribution;
	}

	/**
	 * For a random variable {@code X} whose values are distributed according to
	 * this distribution, this method returns {@code P(X = x)}. In other words,
	 * this method represents the probability mass function (PMF) for the
	 * distribution.
	 *
	 * @param value
	 *            the point at which the PMF is evaluated
	 * @return the value of the probability mass function at point {@code x}
	 */
	// TODO 概率质量函数(probability mass function)是离散型随机变量在各个特定值的概率.
	public double mass(Integer value) {
		return distribution.probability(value);
	}

	@Override
	public double cumulativeDistribution(Integer value) {
		return distribution.cumulativeProbability(value);
	}

	@Override
	public double cumulativeDistribution(Integer minimum, Integer maximum) {
		return distribution.cumulativeProbability(minimum, maximum);
	}

	@Override
	public Integer inverseDistribution(double probability) {
		return distribution.inverseCumulativeProbability(probability);
	}

	@Override
	public Integer sample() {
		return distribution.sample();
	}

	@Override
	public Integer getMaximum() {
		return distribution.getSupportUpperBound();
	}

	@Override
	public Integer getMinimum() {
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
