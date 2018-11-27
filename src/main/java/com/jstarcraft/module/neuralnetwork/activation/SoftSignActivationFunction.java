package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * f_i(x) = x_i / (1+|x_i|)
 */
public class SoftSignActivationFunction implements ActivationFunction {

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = value / (1F + FastMath.abs(value));
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = value / (1F + FastMath.abs(value));
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = 1F + FastMath.abs(value);
			value = 1F / (value * value);
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = 1F + FastMath.abs(value);
			value = 1F / (value * value);
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
		return "SoftSignActivationFunction()";
	}
}
