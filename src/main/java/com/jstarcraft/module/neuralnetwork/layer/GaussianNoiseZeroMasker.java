package com.jstarcraft.module.neuralnetwork.layer;

import org.apache.commons.math3.distribution.NormalDistribution;

import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * Applies additive, mean-zero Gaussian noise to the input - i.e., x = x +
 * N(0,stddev).<br>
 * Note that this differs from {@link GaussianNoiseOneMasker}, which applies
 * <it>multiplicative</it> mean-1 N(1,s) noise.<br>
 * Note also that schedules for the standard deviation value can also be used.
 *
 * @author Alex Black
 */
public class GaussianNoiseZeroMasker implements Masker {

	private Schedule schedule;

	public GaussianNoiseZeroMasker(Schedule schedule) {
		this.schedule = schedule;
	}

	@Override
	public void mask(MathMatrix matrix, int iteration, int epoch) {
		float current = schedule.valueAt(iteration, epoch);

		ContinuousProbability probability = new ContinuousProbability(new NormalDistribution(0F, current));
		matrix.mapValues((row, column, value, message) -> {
			return value + probability.sample().floatValue();
		}, null, MathCalculator.PARALLEL);
	}

}
