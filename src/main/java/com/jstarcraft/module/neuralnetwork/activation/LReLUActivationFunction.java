package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * Leaky RELU f(x) = max(0, x) + alpha * min(0, x) alpha defaults to 0.01
 */
public class LReLUActivationFunction implements ActivationFunction {

	public static final float DEFAULT_ALPHA = 0.01F;

	private float alpha;

	public LReLUActivationFunction() {
		this(DEFAULT_ALPHA);
	}

	public LReLUActivationFunction(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = value < 0F ? alpha * value : value;
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = value < 0F ? alpha * value : value;
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (value >= 0F ? 1F : alpha);
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (value >= 0F ? 1F : alpha);
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
			LReLUActivationFunction that = (LReLUActivationFunction) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.alpha, that.alpha);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(alpha);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "LReLUActivationFunction(alpha=" + alpha + ")";
	}
}
