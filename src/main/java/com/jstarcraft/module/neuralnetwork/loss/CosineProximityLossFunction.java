package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;

/**
 * Created by susaneraly on 9/9/16.
 */
public class CosineProximityLossFunction implements LossFunction {

	@Override
	public float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks) {
		/*
		 * mean of -(y.dot(yhat)/||y||*||yhat||)
		 */
		float score = 0F;
		for (int row = 0; row < trains.getRowSize(); row++) {
			MathVector vector = trains.getRowVector(row);
			MathVector label = tests.getRowVector(row);
			float scoreNorm = Math.max(MathUtility.norm(vector, 2), MathUtility.EPSILON);
			float labelNorm = Math.max(MathUtility.norm(label, 2), MathUtility.EPSILON);
			for (VectorScalar term : vector) {
				score += -(term.getValue() * label.getValue(term.getIndex()) / scoreNorm / labelNorm);
			}
		}
		// TODO 暂时不处理masks
		return score;
	}

	@Override
	public void computeGradient(MathMatrix tests, MathMatrix trains, MathMatrix masks, MathMatrix gradients) {
		for (int row = 0; row < trains.getRowSize(); row++) {
			MathVector vector = trains.getRowVector(row);
			MathVector label = tests.getRowVector(row);
			float scoreNorm = MathUtility.norm(vector, 2);
			float labelNorm = MathUtility.norm(label, 2);
			float squareNorm = scoreNorm * scoreNorm;
			float sum = 0F;
			for (VectorScalar term : vector) {
				sum += term.getValue() * label.getValue(term.getIndex());
			}

			labelNorm = Math.max(labelNorm, MathUtility.EPSILON);
			scoreNorm = Math.max(scoreNorm, MathUtility.EPSILON);
			squareNorm = Math.max(squareNorm, MathUtility.EPSILON);
			for (VectorScalar term : vector) {
				float value = term.getValue();
				value = label.getValue(term.getIndex()) * squareNorm - value * sum;
				value /= (labelNorm * scoreNorm * squareNorm);
				gradients.setValue(row, term.getIndex(), -value);
			}
		}
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
		return "CosineProximityLossFunction()";
	}

}
