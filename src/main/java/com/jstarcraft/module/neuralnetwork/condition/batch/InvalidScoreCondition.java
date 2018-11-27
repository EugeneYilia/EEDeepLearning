package com.jstarcraft.module.neuralnetwork.condition.batch;

import java.util.Map;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.condition.Condition;

/**
 * 无效分数条件
 * 
 * @author Birdy
 *
 */
public class InvalidScoreCondition implements Condition {

	@Override
	public boolean stop(double newScore, double oldScore, Map<String, MathMatrix> gradients) {
		return Double.isNaN(newScore) || Double.isInfinite(newScore);
	}

	@Override
	public String toString() {
		return "InvalidScoreCondition()";
	}
}
