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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;

/**
 * BaseGraphVertex defines a set of common functionality for GraphVertex
 * instances.
 */
@ModelDefinition(value = { "vertexName", "factory" })
public abstract class AbstractVertex implements Vertex {

	protected final Logger logger = LoggerFactory.getLogger(getClass());

	protected String vertexName;

	protected MatrixFactory factory;

	/** 键为inputData(不可以为null),值为outerError(可以为null) */
	protected KeyValue<MathMatrix, MathMatrix>[] inputKeyValues;

	/** 键为outputData(不可以为null),值为innerError(不可以为null) */
	protected KeyValue<MathMatrix, MathMatrix> outputKeyValue;

	protected AbstractVertex() {
	}

	protected AbstractVertex(String name, MatrixFactory factory) {
		this.vertexName = name;
		this.factory = factory;
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		// 检查样本
		if (samples.length == 0) {
			throw new IllegalArgumentException();
		}

		this.inputKeyValues = samples;
		this.outputKeyValue = new KeyValue<>(null, null);
	}

	@Override
	public String getVertexName() {
		return vertexName;
	}

	@Override
	public KeyValue<MathMatrix, MathMatrix> getInputKeyValue(int position) {
		return inputKeyValues[position];
	}

	@Override
	public KeyValue<MathMatrix, MathMatrix> getOutputKeyValue() {
		return outputKeyValue;
	}

	@Override
	public abstract String toString();

}
