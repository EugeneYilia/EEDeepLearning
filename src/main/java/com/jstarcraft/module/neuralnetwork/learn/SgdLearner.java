/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
 *
 */

package com.jstarcraft.module.neuralnetwork.learn;

import java.util.Map;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.schedule.ConstantSchedule;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * SGD updater applies a learning rate only
 * 
 * @author Adam Gibson
 */
@ModelDefinition(value = { "learnSchedule" })
public class SgdLearner implements Learner {

	public static final float DEFAULT_SGD_LR = 1E-3F;

	private Schedule learnSchedule;

	public SgdLearner() {
		this(new ConstantSchedule(DEFAULT_SGD_LR));
	}

	public SgdLearner(Schedule learnSchedule) {
		this.learnSchedule = learnSchedule;
	}

	@Override
	public void doCache(Map<String, MathMatrix> gradients) {
	}

	@Override
	public void learn(Map<String, MathMatrix> gradients, int iteration, int epoch) {
		float learnRatio = learnSchedule.valueAt(iteration, epoch);
		for (MathMatrix gradient : gradients.values()) {
			gradient.scaleValues(learnRatio);
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
			SgdLearner that = (SgdLearner) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.learnSchedule, that.learnSchedule);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(learnSchedule);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "SgdLearner(learnSchedule=" + learnSchedule + ")";
	}

}
