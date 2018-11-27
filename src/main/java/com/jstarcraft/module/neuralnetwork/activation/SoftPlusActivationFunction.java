package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * f(x) = log(1+e^x)
 */
public class SoftPlusActivationFunction implements ActivationFunction {

	private boolean threshold;

	public SoftPlusActivationFunction() {
		this(false);
	}

	public SoftPlusActivationFunction(boolean threshold) {
		this.threshold = threshold;
	}

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (float) FastMath.log(1F + FastMath.exp(value));
			if (threshold && (Float.isNaN(value) || Float.isInfinite(value))) {
				value = MathUtility.EPSILON;
			}
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) FastMath.log(1F + FastMath.exp(value));
			if (threshold && (Float.isNaN(value) || Float.isInfinite(value))) {
				value = MathUtility.EPSILON;
			}
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (float) (1F / (1F + FastMath.exp(-value)));
			if (threshold && (Float.isNaN(value) || Float.isInfinite(value))) {
				value = MathUtility.EPSILON;
			}
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) (1F / (1F + FastMath.exp(-value)));
			if (threshold && (Float.isNaN(value) || Float.isInfinite(value))) {
				value = MathUtility.EPSILON;
			}
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
		return "SoftPlusActivationFunction(threshold=" + threshold + ")";
	}
}
