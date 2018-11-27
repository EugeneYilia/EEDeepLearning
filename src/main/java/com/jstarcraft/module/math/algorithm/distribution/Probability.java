package com.jstarcraft.module.math.algorithm.distribution;

/**
 * 概率分布
 * 
 * @author Birdy
 *
 */
public interface Probability<T extends Number> {

	/**
	 * For a random variable {@code X} whose values are distributed according to
	 * this distribution, this method returns {@code P(X <= x)}. In other words,
	 * this method represents the (cumulative) distribution function (CDF) for
	 * this distribution.
	 *
	 * @param value
	 *            the point at which the CDF is evaluated
	 * @return the probability that a random variable with this distribution
	 *         takes a value less than or equal to {@code x}
	 */
	// TODO 累积分布函数 (cumulative distribution function)
	// 不管是什么类型(连续/离散/其他)的随机变量,都可以定义它的累积分布函数.
	double cumulativeDistribution(T value);

	/**
	 * For a random variable {@code X} whose values are distributed according to
	 * this distribution, this method returns {@code P(x0 < X <= x1)}.
	 *
	 * @param minimum
	 *            the exclusive lower bound
	 * @param maximum
	 *            the inclusive upper bound
	 * @return the probability that a random variable with this distribution
	 *         takes a value between {@code x0} and {@code x1}, excluding the
	 *         lower and including the upper endpoint
	 * @throws org.apache.commons.math3.exception.NumberIsTooLargeException
	 *             if {@code x0 > x1}
	 */
	double cumulativeDistribution(T minimum, T maximum);

	/**
	 * Computes the quantile function of this distribution. For a random
	 * variable {@code X} distributed according to this distribution, the
	 * returned value is
	 * <ul>
	 * <li><code>inf{x in R | P(X<=x) >= p}</code> for {@code 0 < p <= 1},</li>
	 * <li><code>inf{x in R | P(X<=x) > 0}</code> for {@code p = 0}.</li>
	 * </ul>
	 *
	 * @param probability
	 *            the cumulative probability
	 * @return the smallest {@code p}-quantile of this distribution (largest
	 *         0-quantile for {@code p = 0})
	 * @throws org.apache.commons.math3.exception.OutOfRangeException
	 *             if {@code p < 0} or {@code p > 1}
	 */
	T inverseDistribution(double probability);

	/**
	 * Generate a random value sampled from this distribution.
	 *
	 * @return a random value.
	 */
	T sample();

	/**
	 * Access the upper bound of the support. This method must return the same
	 * value as {@code inverseCumulativeProbability(1)}. In other words, this
	 * method must return
	 * <p>
	 * <code>inf {x in R | P(X <= x) = 1}</code>.
	 * </p>
	 *
	 * @return upper bound of the support (might be
	 *         {@code Double.POSITIVE_INFINITY})
	 */
	T getMaximum();

	/**
	 * Access the lower bound of the support. This method must return the same
	 * value as {@code inverseCumulativeProbability(0)}. In other words, this
	 * method must return
	 * <p>
	 * <code>inf {x in R | P(X <= x) > 0}</code>.
	 * </p>
	 *
	 * @return lower bound of the support (might be
	 *         {@code Double.NEGATIVE_INFINITY})
	 */
	T getMinimum();

	/**
	 * Use this method to get the numerical value of the mean of this
	 * distribution.
	 *
	 * @return the mean or {@code Double.NaN} if it is not defined
	 */
	double getMean();

	/**
	 * Use this method to get the numerical value of the variance of this
	 * distribution.
	 *
	 * @return the variance (possibly {@code Double.POSITIVE_INFINITY} as for
	 *         certain cases in
	 *         {@link org.apache.commons.math3.distribution.TDistribution}) or
	 *         {@code Double.NaN} if it is not defined
	 */
	double getVariance();

	void setSeed(long seed);

}
