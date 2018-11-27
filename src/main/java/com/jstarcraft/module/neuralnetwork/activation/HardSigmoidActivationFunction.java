package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * f(x) = min(1, max(0, 0.2*x + 0.5))
 */
public class HardSigmoidActivationFunction implements ActivationFunction {

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = Math.min(1F, Math.max(0F, 0.2F * value + 0.5F));
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = Math.min(1F, Math.max(0F, 0.2F * value + 0.5F));
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (value < -2.5F || value > 2.5F) ? 0F : 0.2F;
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (value < -2.5F || value > 2.5F) ? 0F : 0.2F;
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
		return "HardSigmoidActivationFunction()";
	}
}
