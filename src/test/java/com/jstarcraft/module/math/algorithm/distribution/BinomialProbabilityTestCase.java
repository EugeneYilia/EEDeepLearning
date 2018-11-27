package com.jstarcraft.module.math.algorithm.distribution;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import com.jstarcraft.module.math.algorithm.distribution.DiscreteProbability;
import com.jstarcraft.module.math.algorithm.distribution.Probability;

public class BinomialProbabilityTestCase extends ProbabilityTestCase {

	@Override
	protected Distribution getOldFunction(long seed) {
		Random random = new DefaultRandom(seed);
		Distribution distribution = new org.nd4j.linalg.api.rng.distribution.impl.BinomialDistribution(random, 10, 0.5D);
		return distribution;
	}

	@Override
	protected <T extends Number> Probability<T> getNewFunction(long seed) {
		RandomGenerator random = new SynchronizedRandomGenerator(new MersenneTwister(seed));
		BinomialDistribution distribution = new BinomialDistribution(random, 10, 0.5D);
		return (Probability<T>) new DiscreteProbability(distribution);
	}

	@Override
	protected void assertSample(Probability<Number> newFuction, Distribution oldFunction) {
		Number newSample = newFuction.sample().doubleValue();
		Number oldSample = Math.ceil(oldFunction.sample());
		System.out.println(newSample);
		System.out.println(oldSample);
		Assert.assertThat(newSample, CoreMatchers.equalTo(oldSample));
	}

}
