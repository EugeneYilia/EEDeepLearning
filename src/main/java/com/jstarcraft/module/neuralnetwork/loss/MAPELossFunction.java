package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;

/**
 * Created by susaneraly on 8/15/16.
 */
public class MAPELossFunction implements LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		float score = 0F;
		float scale = 100F / trains.getColumnSize();
		for (MatrixScalar term : trains) {
			double value = term.getValue();
			double label = tests.getValue(term.getRow(), term.getColumn());
			value = (value - label) / label;
			value = Math.abs(value) * scale;
			score += value;
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		float scale = -100F / trains.getColumnSize();
		gradients.mapValues((row, column, value, message) -> {
			value = trains.getValue(row, column);
			float label = tests.getValue(row, column);
			value = label - value;
			value = value < 0F ? -1F : (value > 0F ? 1F : 0F);
			value = value / Math.abs(label) * scale;
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
		return "MAPELossFunction()";
	}

}
