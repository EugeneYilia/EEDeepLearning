/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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

package com.jstarcraft.module.neuralnetwork.vertex;

import java.util.Map;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.deeplearning4j.nn.conf.InputPreProcessor;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.layer.Layer;
import com.jstarcraft.module.neuralnetwork.learn.IgnoreLearner;
import com.jstarcraft.module.neuralnetwork.learn.Learner;
import com.jstarcraft.module.neuralnetwork.normalization.IgnoreNormalizer;
import com.jstarcraft.module.neuralnetwork.normalization.Normalizer;

/**
 * LayerVertex is a GraphVertex with a neural network Layer (and, optionally an
 * {@link InputPreProcessor}) in it
 *
 * @author Alex Black
 */
@ModelDefinition(value = { "vertexName", "factory", "layer", "learner", "normalizer" })
public class LayerVertex extends AbstractVertex {

	protected Layer layer;

	// 梯度相关(自适应学习率)
	protected Learner learner;

	// 梯度相关(归一化)
	protected Normalizer normalizer;

	protected int epoch, iteration;

	protected LayerVertex() {
	}

	public LayerVertex(String name, MatrixFactory factory, Layer layer) {
		this(name, factory, layer, new IgnoreLearner(), new IgnoreNormalizer());
	}

	public LayerVertex(String name, MatrixFactory factory, Layer layer, Learner learner, Normalizer normalizer) {
		super(name, factory);
		this.layer = layer;
		this.learner = learner;
		this.normalizer = normalizer;
		this.epoch = 0;
		this.iteration = 0;
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		layer.doCache(factory, samples[0]);
		learner.doCache(layer.getGradients());
		inputKeyValues = new KeyValue[] { layer.getInputKeyValue() };
		outputKeyValue = layer.getOutputKeyValue();

		epoch++;
		iteration = 0;
	}

	@Override
	public void doForward() {
		layer.doForward();
	}

	@Override
	public void doBackward() {
		layer.doBackward();
		Map<String, MathMatrix> gradients = layer.getGradients();
		// TODO 执行标准器(标准化)
		normalizer.normalize(gradients);
		// 执行学习器(自适应学习率)
		learner.learn(gradients, iteration++, epoch);
	}

	public Layer getLayer() {
		return layer;
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
			LayerVertex that = (LayerVertex) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.vertexName, that.vertexName);
			equal.append(this.layer, that.layer);
			equal.append(this.learner, that.learner);
			equal.append(this.normalizer, that.normalizer);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vertexName);
		hash.append(layer);
		hash.append(learner);
		hash.append(normalizer);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "LayerVertex(name=" + vertexName + ")";
	}

}
