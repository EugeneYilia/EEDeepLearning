package com.jstarcraft.module.neuralnetwork.condition.iterate;

import java.util.Map;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.condition.Condition;

/**
 * Epsilon termination (absolute change based on tolerance)
 *
 * @author Adam Gibson
 */
public class EpsilonCondition implements Condition {

	private double epsilon = MathUtility.EPSILON;

	private double tolerance = MathUtility.EPSILON;

	public EpsilonCondition() {
	}

	public EpsilonCondition(double epsilon, double tolerance) {
		this.epsilon = epsilon;
		this.tolerance = tolerance;
	}

	@Override
	public boolean stop(double newScore, double oldScore, Map<String, MathMatrix> gradients) {
		// special case for initial termination, ignore
		if (newScore == 0D && oldScore == 0D) {
			return false;
		}

		double oldAbsolute = FastMath.abs(oldScore);
		double newAbsolute = FastMath.abs(newScore);

		return 2D * FastMath.abs(oldScore - newScore) <= tolerance * (oldAbsolute + newAbsolute + epsilon);
	}

}
