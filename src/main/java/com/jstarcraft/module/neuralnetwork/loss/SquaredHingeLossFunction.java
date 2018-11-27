package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;

/**
 * Created by susaneraly on 9/9/16.
 */
public class SquaredHingeLossFunction implements LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		/*
		 * y_hat is -1 or 1 hinge loss is max(0,1-y_hat*y)
		 */
		float score = 0F;
		for (MatrixScalar term : trains) {
			float value = term.getValue();
			value = 1F - value * tests.getValue(term.getRow(), term.getColumn());
			// value = Math.max(0D, value);
			score += value * value;
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		/*
		 * bit mask is 0 if 1-sigma(y*yhat) is neg bit mask is 1 if 1-sigma(y*yhat) is
		 * +ve
		 */
		gradients.mapValues((row, column, value, message) -> {
			value = trains.getValue(row, column);
			value = 1F - value * tests.getValue(row, column);
			float left = -tests.getValue(row, column) * value * 2;
			float right = value > 0F ? 1F : 0F;
			return left * right;
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
		return "SquaredHingeLossFunction()";
	}

}
