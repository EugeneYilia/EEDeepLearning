package com.jstarcraft.module.neuralnetwork.loss;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 损失函数
 * 
 * @author Birdy
 *
 */
public interface LossFunction {

	default void doCache(MathMatrix tests, MathMatrix trains) {
	}

	/**
	 * Compute the score (loss function value) for each example individually.
	 * For input [numExamples,nOut] returns scores as a column vector:
	 * [numExamples,1]
	 * 
	 * @param tests
	 *            Labels/expected output
	 * @param trains
	 *            Output of the model (neural network)
	 * @param activationFn
	 *            Activation function that should be applied to preOutput
	 * @param masks
	 * @return Loss function value for each example; column vector
	 */
	float computeScore(MathMatrix tests, MathMatrix trains, MathMatrix masks);

	/**
	 * Compute the gradient of the loss function with respect to the inputs:
	 * dL/dOutput
	 *
	 * @param tests
	 *            Label/expected output
	 * @param trains
	 *            Output of the model (neural network), before the activation
	 *            function is applied
	 * @param activationFn
	 *            Activation function that should be applied to preOutput
	 * @param masks
	 *            Mask array; may be null
	 * @return Gradient dL/dPreOut
	 */
	void computeGradient(MathMatrix tests, MathMatrix trains,MathMatrix masks, MathMatrix gradients);

}
