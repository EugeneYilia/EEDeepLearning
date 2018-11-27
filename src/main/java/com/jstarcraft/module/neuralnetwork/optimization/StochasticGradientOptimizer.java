/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package com.jstarcraft.module.neuralnetwork.optimization;

import java.util.Map;
import java.util.concurrent.Callable;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.condition.Condition;
import com.jstarcraft.module.neuralnetwork.step.NegativeGradientStepFunction;
import com.jstarcraft.module.neuralnetwork.step.StepFunction;

/**
 * Stochastic Gradient Descent Standard fix step size No line search
 * 
 * @author Adam Gibson
 */
public class StochasticGradientOptimizer extends AbstractOptimizer {

	protected Map<String, MathMatrix> gradients, parameters;

	protected StochasticGradientOptimizer() {
	}

	public StochasticGradientOptimizer(Condition... conditions) {
		this(new NegativeGradientStepFunction(), conditions);
	}

	public StochasticGradientOptimizer(StepFunction stepFunction, Condition... terminationConditions) {
		super(stepFunction, terminationConditions);
	}

	@Override
	public void doCache(Callable<Float> scorer, Map<String, MathMatrix> gradients, Map<String, MathMatrix> parameters) {
		this.gradients = gradients;
		this.parameters = parameters;
	}

	@Override
	public boolean optimize(float score) {
		oldScore = newScore;
		newScore = score;

		// 使用梯度更新参数
		stepFunction.step(1F, gradients, parameters);

		for (Condition condition : conditions) {
			if (condition.stop(newScore, oldScore, gradients)) {
				return true;
			}
		}
		return false;
	}

}
