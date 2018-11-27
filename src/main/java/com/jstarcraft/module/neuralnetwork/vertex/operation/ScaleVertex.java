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

package com.jstarcraft.module.neuralnetwork.vertex.operation;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.AbstractVertex;

/**
 * A ScaleVertex is used to scale the size of activations of a single layer<br>
 * For example, ResNet activations can be scaled in repeating blocks to keep
 * variance under control.
 *
 * @author Justin Long (@crockpotveggies)
 */
@ModelDefinition(value = { "vertexName", "factory", "scaleFactor" })
public class ScaleVertex extends AbstractVertex {

	private float scaleFactor;

	protected ScaleVertex() {
	}

	public ScaleVertex(String name, MatrixFactory factory, float scaleFactor) {
		super(name, factory);
		this.scaleFactor = scaleFactor;
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		super.doCache(samples);

		// 检查样本的数量是否一样
		int rowSize = samples[0].getKey().getRowSize();
		for (int position = 1; position < samples.length; position++) {
			if (rowSize != samples[position].getKey().getRowSize()) {
				throw new IllegalArgumentException();
			}
		}

		// 检查样本的维度是否一样
		int columnSize = samples[0].getKey().getColumnSize();
		for (int position = 1; position < samples.length; position++) {
			if (columnSize != samples[position].getKey().getColumnSize()) {
				throw new IllegalArgumentException();
			}
		}

		// TODO 考虑支持CompositeMatrix.
		MathMatrix outputData = factory.makeCache(rowSize, columnSize);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = factory.makeCache(rowSize, columnSize);
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
		MathMatrix inputData = inputKeyValues[0].getKey();
		MathMatrix outputData = outputKeyValue.getKey();
		outputData.mapValues((row, column, value, message) -> {
			value = inputData.getValue(row, column) * scaleFactor;
			return value;
		}, null, MathCalculator.PARALLEL);
		MathMatrix innerError = outputKeyValue.getValue();
		innerError.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
	}

	@Override
	public void doBackward() {
		MathMatrix innerError = outputKeyValue.getValue();
		MathMatrix outerError = inputKeyValues[0].getValue();
		if (outerError != null) {
			outerError.mapValues((row, column, value, message) -> {
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				value = value + innerError.getValue(row, column) * scaleFactor;
				return value;
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
			ScaleVertex that = (ScaleVertex) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.vertexName, that.vertexName);
			equal.append(this.scaleFactor, that.scaleFactor);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vertexName);
		hash.append(scaleFactor);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "ScaleVertex(name=" + vertexName + ", scale=" + scaleFactor + ")";
	}

}
