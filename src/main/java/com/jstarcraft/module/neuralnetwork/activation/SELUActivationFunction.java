package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.neuralnetwork.layer.AlphaMasker;

/**
 * {@link AlphaMasker} https://arxiv.org/pdf/1706.02515.pdf
 */
public class SELUActivationFunction implements ActivationFunction {

	private static final float SELU_ALPHA = 1.6732632423543772848170429916717F;

	private static final float SELU_LAMBDA = 1.0507009873554804934193349852946F;

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (float) (value > 0F ? SELU_LAMBDA * value : SELU_LAMBDA * (SELU_ALPHA * FastMath.exp(value) - SELU_ALPHA));
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) (value > 0F ? SELU_LAMBDA * value : SELU_LAMBDA * (SELU_ALPHA * FastMath.exp(value) - SELU_ALPHA));
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (float) (value > 0F ? SELU_LAMBDA : SELU_ALPHA * SELU_LAMBDA * FastMath.exp(value));
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) (value > 0F ? SELU_LAMBDA : SELU_ALPHA * SELU_LAMBDA * FastMath.exp(value));
			value *= error.getValue(index);
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			return true;
		}
	}

	@Override
	public int hashCode() {
		return getClass().hashCode();
	}

	@Override
	public String toString() {
		return "SELUActivationFunction()";
	}

}
