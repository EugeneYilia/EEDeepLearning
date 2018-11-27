package com.jstarcraft.module.neuralnetwork.activation;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
public class TanHActivationFunction implements ActivationFunction {

	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		if (input instanceof Nd4jMatrix && output instanceof Nd4jMatrix) {
			INDArray inputArray = Nd4jMatrix.class.cast(input).getArray();
			INDArray outputArray = Nd4jMatrix.class.cast(output).getArray();
			Nd4j.getExecutioner().execAndReturn(new Tanh(inputArray, outputArray));
		} else {
			output.mapValues((row, column, value, message) -> {
				value = input.getValue(row, column);
				value = (float) FastMath.tanh(value);
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) FastMath.tanh(value);
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		if (input instanceof Nd4jMatrix && output instanceof Nd4jMatrix && error instanceof Nd4jMatrix) {
			INDArray inputArray = Nd4jMatrix.class.cast(input).getArray();
			INDArray outputArray = Nd4jMatrix.class.cast(output).getArray();
			INDArray errorArray = Nd4jMatrix.class.cast(error).getArray();
			Nd4j.getExecutioner().execAndReturn(new TanhDerivative(inputArray, outputArray));
			outputArray.muli(errorArray);
		} else {
			output.mapValues((row, column, value, message) -> {
				value = input.getValue(row, column);
				value = (float) FastMath.tanh(value);
				value = 1F - value * value;
				value *= error.getValue(row, column);
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (float) FastMath.tanh(value);
			value = 1F - value * value;
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
		return "TanHActivationFunction()";
	}

}
