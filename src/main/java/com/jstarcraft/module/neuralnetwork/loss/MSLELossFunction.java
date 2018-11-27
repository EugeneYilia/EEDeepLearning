package com.jstarcraft.module.neuralnetwork.loss;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;

/**
 * Mean Squared Logarithmic Error loss function: L = 1/N sum_i
 * (log(1+predicted_i) - log(1+actual_i))^2
 *
 * @author Susan Eraly
 */
public class MSLELossFunction implements LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		float score = 0F;
		float scale = trains.getColumnSize();
		for (MatrixScalar term : trains) {
			float value = term.getValue();
			value = (float) (FastMath.log((value + 1F) / (tests.getValue(term.getRow(), term.getColumn()) + 1F)));
			value = value * value / scale;
			score += value;
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		float scale = 2F / trains.getColumnSize();
		gradients.mapValues((row, column, value, message) -> {
			value = trains.getValue(row, column);
			float ratio = (float) (FastMath.log((value + 1F) / (tests.getValue(row, column) + 1F)));
			value = scale / (value + 1F);
			return value * ratio;
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
		return "MSLELossFunction()";
	}

}
