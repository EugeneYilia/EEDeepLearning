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

package com.jstarcraft.module.neuralnetwork.step;

import java.util.Map;
import java.util.Map.Entry;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * Subtract the line
 *
 * @author Adam Gibson
 */
public class NegativeGradientStepFunction implements StepFunction {

	@Override
	public void step(float step, Map<String, MathMatrix> directions, Map<String, MathMatrix> parameters) {
		// TODO 考虑优化性能
		for (Entry<String, MathMatrix> keyValue : parameters.entrySet()) {
			MathMatrix parameter = keyValue.getValue();
			MathMatrix direction = directions.get(keyValue.getKey());
			parameter.mapValues((row, column, value, message) -> {
				return value - direction.getValue(row, column);
			}, null, MathCalculator.PARALLEL);
		}
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
		return "NegativeGradientStepFunction()";
	}

}
