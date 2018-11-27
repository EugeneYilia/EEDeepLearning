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

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.model.ModelCycle;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.schedule.ConstantSchedule;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * The Adam updater. http://arxiv.org/abs/1412.6980
 *
 * @author Adam Gibson
 */
@ModelDefinition(value = { "beta1", "beta2", "epsilon", "learnSchedule" })
public class AdamLearner implements Learner, ModelCycle {

	public static final float DEFAULT_ADAM_LEARN_RATE = 1E-3F;
	public static final float DEFAULT_ADAM_EPSILON = 1E-8F;
	public static final float DEFAULT_ADAM_BETA1_MEAN_DECAY = 0.9F;
	public static final float DEFAULT_ADAM_BETA2_VAR_DECAY = 0.999F;

	// gradient moving avg decay rate
	private float beta1;
	// gradient sqrd decay rate
	private float beta2;
	private float epsilon;

	private Schedule learnSchedule;

	// moving avg & sqrd gradients
	private Map<String, DenseMatrix> ms, vs;

	public AdamLearner() {
		this(DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON, new ConstantSchedule(DEFAULT_ADAM_LEARN_RATE));
	}

	public AdamLearner(Schedule learnSchedule) {
		this(DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON, learnSchedule);
	}

	public AdamLearner(float beta1, float beta2, float epsilon, Schedule learnSchedule) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
		this.learnSchedule = learnSchedule;
		this.ms = new HashMap<>();
		this.vs = new HashMap<>();
	}

	@Override
	public void doCache(Map<String, MathMatrix> gradients) {
		ms.clear();
		vs.clear();
		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			ms.put(term.getKey(), DenseMatrix.valueOf(gradient.getRowSize(), gradient.getColumnSize()));
			vs.put(term.getKey(), DenseMatrix.valueOf(gradient.getRowSize(), gradient.getColumnSize()));
		}
	}

	@Override
	public void learn(Map<String, MathMatrix> gradients, int iteration, int epoch) {
		if (ms.isEmpty() || vs.isEmpty()) {
			throw new IllegalStateException("Updater has not been initialized with view state");
		}

		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			DenseMatrix m = ms.get(term.getKey());
			DenseMatrix v = vs.get(term.getKey());

			m.mapValues((row, column, value, message) -> {
				float delta = gradient.getValue(row, column);
				value = value * beta1 + delta * (1F - beta1);
				return value;
			}, null, MathCalculator.PARALLEL);

			v.mapValues((row, column, value, message) -> {
				float delta = gradient.getValue(row, column);
				value = value * beta2 + delta * delta * (1F - beta2);
				return value;
			}, null, MathCalculator.PARALLEL);

			float beta1t = (float) FastMath.pow(beta1, iteration + 1);
			float beta2t = (float) FastMath.pow(beta2, iteration + 1);
			float learnRatio = learnSchedule.valueAt(iteration, epoch);
			float alphat = (float) (learnRatio * FastMath.sqrt(1F - beta2t) / (1F - beta1t));
			if (Double.isNaN(alphat) || alphat == 0D) {
				alphat = epsilon;
			}

			float alpha = alphat;
			gradient.mapValues((row, column, value, message) -> {
				value = (float) (m.getValue(row, column) * alpha / (FastMath.sqrt(v.getValue(row, column)) + epsilon));
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public void beforeSave() {
	}

	@Override
	public void afterLoad() {
		ms = new HashMap<>();
		vs = new HashMap<>();
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
			AdamLearner that = (AdamLearner) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.beta1, that.beta1);
			equal.append(this.beta2, that.beta2);
			equal.append(this.epsilon, that.epsilon);
			equal.append(this.learnSchedule, that.learnSchedule);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(beta1);
		hash.append(beta2);
		hash.append(epsilon);
		hash.append(learnSchedule);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "AdamLearner(beta1=" + beta1 + ", beta2=" + beta2 + ", epsilon=" + epsilon + ", learnSchedule=" + learnSchedule + ")";
	}

}
