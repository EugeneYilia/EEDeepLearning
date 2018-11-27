package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.message.SumMessage;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;

/**
 * f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift) where shift = max_i(x_i)
 */
public class SoftmaxActivationFunction implements ActivationFunction {

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		for (int row = 0; row < output.getRowSize(); row++) {
			double maximum = Double.NEGATIVE_INFINITY;
			MathVector inputVector = input.getRowVector(row);
			for (VectorScalar term : inputVector) {
				maximum = FastMath.max(maximum, term.getValue());
			}
			double shift = maximum;
			SumMessage sum = new SumMessage(false);
			MathVector outputVector = output.getRowVector(row);
			outputVector.mapValues((index, value, message) -> {
				value = inputVector.getValue(index);
				value = (float) FastMath.exp(value - shift);
				message.accumulateValue(value);
				return value;
			}, sum, MathCalculator.SERIAL);
			outputVector.mapValues((index, value, message) -> {
				return value / sum.getValue();
			}, sum, MathCalculator.SERIAL);
		}
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		double maximum = Double.NEGATIVE_INFINITY;
		for (VectorScalar term : input) {
			maximum = FastMath.max(maximum, term.getValue());
		}
		double shift = maximum;
		SumMessage sum = new SumMessage(false);
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) FastMath.exp(value - shift);
			message.accumulateValue(value);
			return value;
		}, sum, MathCalculator.SERIAL);
		output.mapValues((index, value, message) -> {
			return value / sum.getValue();
		}, sum, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		{
			for (int row = 0; row < output.getRowSize(); row++) {
				double maximum = Double.NEGATIVE_INFINITY;
				MathVector vector = input.getRowVector(row);
				for (VectorScalar term : vector) {
					maximum = FastMath.max(maximum, term.getValue());
				}
				double shift = maximum;
				SumMessage sum = new SumMessage(false);
				MathVector outputVector = output.getRowVector(row);
				outputVector.mapValues((index, value, message) -> {
					value = vector.getValue(index);
					value = (float) FastMath.exp(value - shift);
					message.accumulateValue(value);
					return value;
				}, sum, MathCalculator.SERIAL);
				outputVector.mapValues((index, value, message) -> {
					return value / sum.getValue();
				}, sum, MathCalculator.SERIAL);
			}
		}

		{
			for (int row = 0; row < output.getRowSize(); row++) {
				MathVector vector = output.getRowVector(row);
				float sum = 0F;
				for (VectorScalar term : vector) {
					sum += term.getValue() * error.getValue(row, term.getIndex());
				}
				float shift = sum;
				for (VectorScalar term : vector) {
					float value = term.getValue();
					value *= (error.getValue(row, term.getIndex()) - shift);
					term.setValue(value);
				}
			}
		}
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		{
			double maximum = Double.NEGATIVE_INFINITY;
			for (VectorScalar term : input) {
				maximum = FastMath.max(maximum, term.getValue());
			}
			double shift = maximum;
			SumMessage sum = new SumMessage(false);
			output.mapValues((index, value, message) -> {
				value = input.getValue(index);
				value = (float) FastMath.exp(value - shift);
				message.accumulateValue(value);
				return value;
			}, sum, MathCalculator.SERIAL);
			output.mapValues((index, value, message) -> {
				return value / sum.getValue();
			}, sum, MathCalculator.SERIAL);
		}

		{
			float sum = 0F;
			for (VectorScalar term : output) {
				sum += term.getValue() * error.getValue(term.getIndex());
			}
			float shift = sum;
			for (VectorScalar term : output) {
				float value = term.getValue();
				value *= (error.getValue(term.getIndex()) - shift);
				term.setValue(value);
			}
		}
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
		return "SoftmaxActivationFunction()";
	}

}
