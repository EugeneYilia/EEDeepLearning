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
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelCycle;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.schedule.ConstantSchedule;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * RMS Prop updates:
 * <p>
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@ModelDefinition(value = { "rmsDecay", "epsilon", "learnSchedule" })
public class RmsPropLearner implements Learner, ModelCycle {

	public static final float DEFAULT_RMSPROP_LEARN_RATE = 1E-1F;
	public static final float DEFAULT_RMSPROP_EPSILON = 1E-8F;
	public static final float DEFAULT_RMSPROP_RMSDECAY = 0.95F;

	private float rmsDecay;
	private float epsilon;

	private Schedule learnSchedule;

	private Map<String, DenseMatrix> lastGradients;

	public RmsPropLearner() {
		this(DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON, new ConstantSchedule(DEFAULT_RMSPROP_LEARN_RATE));
	}

	public RmsPropLearner(float rmsDecay, float epsilon, Schedule learnSchedule) {
		this.rmsDecay = rmsDecay;
		this.epsilon = epsilon;
		this.learnSchedule = learnSchedule;
		this.lastGradients = new HashMap<>();
	}

	@Override
	public void doCache(Map<String, MathMatrix> gradients) {
		lastGradients.clear();
		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			lastGradients.put(term.getKey(), DenseMatrix.valueOf(gradient.getRowSize(), gradient.getColumnSize(), MatrixMapper.constantOf(epsilon)));
		}
	}

	@Override
	public void learn(Map<String, MathMatrix> gradients, int iteration, int epoch) {
		if (lastGradients.isEmpty()) {
			throw new IllegalStateException("Updater has not been initialized with view state");
		}
		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			DenseMatrix lastGradient = lastGradients.get(term.getKey());

			double learnRatio = learnSchedule.valueAt(iteration, epoch);

			lastGradient.mapValues((row, column, value, message) -> {
				float delta = gradient.getValue(row, column);
				value = value * rmsDecay + delta * delta * (1F - rmsDecay);
				return value;
			}, null, MathCalculator.PARALLEL);

			// lr * gradient / (sqrt(cache) + 1e-8)
			gradient.mapValues((row, column, value, message) -> {
				value = (float) (value * (learnRatio / (FastMath.sqrt(lastGradient.getValue(row, column)) + epsilon)));
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public void beforeSave() {
	}

	@Override
	public void afterLoad() {
		lastGradients = new HashMap<>();
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
			RmsPropLearner that = (RmsPropLearner) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.rmsDecay, that.rmsDecay);
			equal.append(this.epsilon, that.epsilon);
			equal.append(this.learnSchedule, that.learnSchedule);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(rmsDecay);
		hash.append(epsilon);
		hash.append(learnSchedule);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "RmsPropLearner(rmsDecay=" + rmsDecay + ", epsilon=" + epsilon + ", learnSchedule=" + learnSchedule + ")";
	}

}
