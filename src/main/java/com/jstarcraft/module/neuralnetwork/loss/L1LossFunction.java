package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;

/**
 * L1 loss function: i.e., sum of absolute errors, L = sum_i abs(predicted_i -
 * actual_i) See also {@link MAELossFunction} for a mathematically similar loss
 * function (MAE has division by N, where N is output size)
 *
 * @author Susan Eraly
 */
public class L1LossFunction implements LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		float score = 0F;
		for (MatrixScalar term : trains) {
			float value = term.getValue();
			value = value - tests.getValue(term.getRow(), term.getColumn());
			score += Math.abs(value);
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		gradients.mapValues((row, column, value, message) -> {
			value = trains.getValue(row, column);
			value = value - tests.getValue(row, column);
			value = value < 0F ? -1F : (value > 0F ? 1F : 0F);
			return value;
		}, null, MathCalculator.PARALLEL);
		// TODO 暂时不处理masks
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
		return "L1LossFunction";
	}

}
