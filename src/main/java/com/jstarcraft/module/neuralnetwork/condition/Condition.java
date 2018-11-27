package com.jstarcraft.module.neuralnetwork.condition;

import java.util.Map;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 条件
 * 
 * @author Birdy
 *
 */
public interface Condition {

	/**
	 * 启动
	 */
	default void start() {
	}

	/**
	 * 停止
	 * 
	 * @param newScore
	 * @param oldScore
	 * @param gradients
	 * @return
	 */
	boolean stop(double newScore, double oldScore, Map<String, MathMatrix> gradients);

}
