package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * f(x) = alpha * (exp(x) - 1.0); x < 0 = x ; x>= 0
 *
 * alpha defaults to 1, if not specified
 */
public class ELUActivationFunction implements ActivationFunction {

	public static final float DEFAULT_ALPHA = 1F;

	private float alpha;

	public ELUActivationFunction() {
		this(DEFAULT_ALPHA);
	}

	public ELUActivationFunction(float alpha) {
		this.alpha = alpha;
	}

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			if (value < 0D) {
				value = (float) ((FastMath.exp(value) - 1D) * alpha);
			}
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			if (value < 0D) {
				value = (float) ((FastMath.exp(value) - 1D) * alpha);
			}
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = value >= 0F ? 1F : (float) (FastMath.exp(value) * alpha);
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = value >= 0F ? 1F : (float) (FastMath.exp(value) * alpha);
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
			ELUActivationFunction that = (ELUActivationFunction) object;
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
		return "ELUActivationFunction(alpha=" + alpha + ")";
	}
}
