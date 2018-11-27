package com.jstarcraft.module.neuralnetwork.layer;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.activation.SELUActivationFunction;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * AlphaDropout is a dropout technique proposed by Klaumbauer et al. 2017 -
 * Self-Normalizing Neural Networks <a href=
 * "https://arxiv.org/abs/1706.02515">https://arxiv.org/abs/1706.02515</a> <br>
 * <br>
 * This dropout technique was designed specifically for self-normalizing neural
 * networks - i.e., networks using
 * {@link org.nd4j.linalg.activations.impl.ActivationSELU} /
 * {@link org.nd4j.linalg.activations.Activation#SELU} activation function,
 * combined with the N(0,stdev=1/sqrt(fanIn)) "SNN" weight initialization,
 * {@link org.deeplearning4j.nn.weights.WeightInit#NORMAL}<br>
 * <br>
 * In conjuction with the aforementioned activation function and weight
 * initialization, AlphaDropout attempts to keep both the mean and variance of
 * the post-dropout activations to the the same (in expectation) as before alpha
 * dropout was applied.<br>
 * Specifically, AlphaDropout implements a * (x * d + alphaPrime * (1-d)) + b,
 * where d ~ Bernoulli(p), i.e., d \in {0,1}. Where x is the input activations,
 * a, b, alphaPrime are constants determined from the SELU alpha/lambda
 * parameters. Users should use the default alpha/lambda values in virtually all
 * cases.<br>
 * <br>
 * Dropout schedules (i.e., varying probability p as a function of
 * iteration/epoch) are also supported.<br>
 * 
 * {@link SELUActivationFunction}
 *
 * @author Alex Black
 */
public class AlphaMasker implements Masker {

	public static final float DEFAULT_ALPHA = 1.6732632423543772F;
	public static final float DEFAULT_LAMBDA = 1.0507009873554804F;

	private Schedule schedule;

	private Float value;
	private float prime;
	private float alpha;
	private float beta;

	public AlphaMasker(Schedule schedule) {
		this(schedule, DEFAULT_ALPHA, DEFAULT_LAMBDA);
	}

	public AlphaMasker(Schedule schedule, float alpha, float lambda) {
		this.schedule = schedule;
		this.prime = -lambda * alpha;
	}

	private float alpha(float probability) {
		return (float) (1F / Math.sqrt(probability + prime * prime * probability * (1F - probability)));
	}

	private float beta(float probability) {
		return -alpha(probability) * (1F - probability) * prime;
	}

	@Override
	public void mask(MathMatrix matrix, int iteration, int epoch) {
		// https://arxiv.org/pdf/1706.02515.pdf pg6
		// "...we propose “alpha dropout”, that randomly sets inputs to α'"
		// "The affine transformation a(xd + α'(1−d))+b allows to determine
		// parameters a and b such that mean and
		// variance are kept to their values"

		float probability = schedule.valueAt(iteration, epoch);
		if (probability != value) {
			alpha = alpha(probability);
			beta = beta(probability);
		}
		value = probability;

		// 参考Pytorch实现.
		// https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
		matrix.mapValues((row, column, value, message) -> {
			value = RandomUtility.randomFloat(1F) < probability ? prime : value;
			return value * alpha + beta;
		}, null, MathCalculator.PARALLEL);
	}

}
