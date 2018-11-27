package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * Mean Squared Error loss function: L = 1/N sum_i (actual_i - predicted)^2 See
 * also {@link L2LossFunction} for a mathematically similar loss function
 * (LossL2 does not have division by N, where N is output size)
 *
 * @author Susan Eraly
 */
public class MSELossFunction extends L2LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		float score = super.computeScore(tests, trains, masks);
		return score / trains.getColumnSize();
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		super.computeGradient(tests, trains, masks, gradients);
		float scale = 1F / trains.getColumnSize();
		gradients.scaleValues(scale);
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
		return "MSELossFunction()";
	}

}
