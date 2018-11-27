package com.jstarcraft.module.neuralnetwork.layer;

import org.apache.commons.math3.distribution.NormalDistribution;

import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

import lombok.Data;

/**
 * Gaussian dropout. This is a multiplicative Gaussian noise (mean 1) on the
 * input activations.<br>
 * <br>
 * Each input activation x is independently set to:<br>
 * x <- x * y, where y ~ N(1, stdev = sqrt((1-rate)/rate))<br>
 * Dropout schedules (i.e., varying probability p as a function of
 * iteration/epoch) are also supported.<br>
 * <br>
 * Note 1: As per all IDropout instances, GaussianDropout is applied at training
 * time only - and is automatically not applied at test time (for evaluation,
 * etc)<br>
 * Note 2: Frequently, dropout is not applied to (or, has higher retain
 * probability for) input (first layer) layers. Dropout is also often not
 * applied to output layers.<br>
 * <br>
 * See: "Multiplicative Gaussian Noise" in Srivastava et al. 2014: Dropout: A
 * Simple Way to Prevent Neural Networks from Overfitting <a href=
 * "http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf">http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf</a>
 *
 * @author Alex Black
 */
@Data
public class GaussianNoiseOneMasker implements Masker {

	private Schedule schedule;

	protected GaussianNoiseOneMasker(Schedule schedule) {
		this.schedule = schedule;
	}

	@Override
	public void mask(MathMatrix matrix, int iteration, int epoch) {
		float current = schedule.valueAt(iteration, epoch);
		current = (float) Math.sqrt(current / (1F - current));

		ContinuousProbability probability = new ContinuousProbability(new NormalDistribution(1F, current));
		matrix.mapValues((row, column, value, message) -> {
			return value * probability.sample().floatValue();
		}, null, MathCalculator.PARALLEL);
	}

}
