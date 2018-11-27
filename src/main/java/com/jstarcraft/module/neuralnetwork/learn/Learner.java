package com.jstarcraft.module.neuralnetwork.learn;

import java.util.Map;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * Gradient modifications: Calculates an update and tracks related information
 * for gradient changes over time for handling updates.
 *
 * @author Alex Black
 */
public interface Learner {

	/**
	 * 根据指定的梯度分配缓存(每次epoch调用)
	 * 
	 * @param numberOfInstances
	 * @param numberOfParameters
	 */
	void doCache(Map<String, MathMatrix> gradients);

	/**
	 * Modify the gradient to be an update. Note that this is be done in-place
	 *
	 * @param gradient
	 *            the gradient to modify
	 * @param iteration
	 * @return the modified gradient
	 */
	void learn(Map<String, MathMatrix> gradients, int iteration, int epoch);

}
