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

/**
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 * https://arxiv.org/pdf/1212.5701v1.pdf
 * <p>
 * Ada delta updater. More robust adagrad that keeps track of a moving window
 * average of the gradient rather than the every decaying learning rates of
 * adagrad
 *
 * @author Adam Gibson
 */
@ModelDefinition(value = { "rho", "epsilon" })
public class AdaDeltaLearner implements Learner, ModelCycle {

	public static final float DEFAULT_ADADELTA_RHO = 0.95F;
	public static final float DEFAULT_ADADELTA_EPSILON = 1E-6F;

	private float rho;
	private float epsilon;

	private Map<String, DenseMatrix> msgs; // E[g^2]_t by arxiv paper, algorithm
											// 1
	private Map<String, DenseMatrix> msdxes; // E[delta x^2]_t by arxiv paper,
												// algorithm 1

	public AdaDeltaLearner() {
		this(DEFAULT_ADADELTA_RHO, DEFAULT_ADADELTA_EPSILON);
	}

	public AdaDeltaLearner(float rho, float epsilon) {
		this.rho = rho;
		this.epsilon = epsilon;
		this.msgs = new HashMap<>();
		this.msdxes = new HashMap<>();
	}

	@Override
	public void doCache(Map<String, MathMatrix> gradients) {
		msgs.clear();
		msdxes.clear();
		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			msgs.put(term.getKey(), DenseMatrix.valueOf(gradient.getRowSize(), gradient.getColumnSize()));
			msdxes.put(term.getKey(), DenseMatrix.valueOf(gradient.getRowSize(), gradient.getColumnSize()));
		}
	}

	@Override
	public void learn(Map<String, MathMatrix> gradients, int iteration, int epoch) {
		if (msgs.isEmpty() || msdxes.isEmpty()) {
			throw new IllegalStateException("Updater has not been initialized with view state");
		}

		for (Entry<String, MathMatrix> term : gradients.entrySet()) {
			MathMatrix gradient = term.getValue();
			DenseMatrix msg = msgs.get(term.getKey());
			DenseMatrix msdx = msdxes.get(term.getKey());

			// Line 4 of Algorithm 1: https://arxiv.org/pdf/1212.5701v1.pdf
			// E[g^2]_t = rho * E[g^2]_{t−1} + (1-rho)*g^2_t
			msg.mapValues((row, column, value, message) -> {
				float delta = gradient.getValue(row, column);
				value = value * rho + delta * delta * (1F - rho);
				return value;
			}, null, MathCalculator.PARALLEL);

			// Calculate update:
			// dX = - g * RMS[delta x]_{t-1} / RMS[g]_t
			// Note: negative is applied in the DL4J step function: params -=
			// update
			// rather than params += update
			gradient.mapValues((row, column, value, message) -> {
				value = (float) (value * (FastMath.sqrt(msdx.getValue(row, column) + epsilon) / FastMath.sqrt(msg.getValue(row, column) + epsilon)));
				return value;
			}, null, MathCalculator.PARALLEL);

			// Accumulate gradients: E[delta x^2]_t = rho * E[delta x^2]_{t-1} +
			// (1-rho)* (delta x_t)^2
			msdx.mapValues((row, column, value, message) -> {
				float delta = gradient.getValue(row, column);
				value = value * rho + delta * delta * (1F - rho);
				return value;
			}, null, MathCalculator.PARALLEL);
		}
	}

	@Override
	public void beforeSave() {
	}

	@Override
	public void afterLoad() {
		msgs = new HashMap<>();
		msdxes = new HashMap<>();
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
			AdaDeltaLearner that = (AdaDeltaLearner) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.rho, that.rho);
			equal.append(this.epsilon, that.epsilon);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(rho);
		hash.append(epsilon);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "AdaDeltaLearner(rho=" + rho + ", epsilon=" + epsilon + ")";
	}

}
