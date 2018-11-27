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

package com.jstarcraft.module.neuralnetwork.vertex.accumulation;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.AbstractVertex;

/**
 * L2Vertex calculates the L2 least squares error of two inputs.
 *
 * For example, in Triplet Embedding you can input an anchor and a pos/neg class
 * and use two parallel L2 vertices to calculate two real numbers which can be
 * fed into a LossLayer to calculate TripletLoss.
 *
 * @author Justin Long (crockpotveggies)
 */
@ModelDefinition(value = { "vertexName", "factory", "epsilon" })
public class EuclideanVertex extends AbstractVertex {

	public static final float DEFAULT_EPSILON = 1E-5F;

	private float epsilon;

	private int columnSize;

	private MathMatrix differences;

	protected EuclideanVertex() {
	}

	public EuclideanVertex(String name, MatrixFactory factory) {
		this(name, factory, DEFAULT_EPSILON);
	}

	public EuclideanVertex(String name, MatrixFactory factory, float epsilon) {
		super(name, factory);
		this.epsilon = epsilon;
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
		columnSize = samples[0].getKey().getColumnSize();
		for (int position = 1; position < samples.length; position++) {
			if (columnSize != samples[position].getKey().getColumnSize()) {
				throw new IllegalArgumentException();
			}
		}

		MathMatrix outputData = factory.makeCache(rowSize, 1);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = factory.makeCache(rowSize, 1);
		outputKeyValue.setValue(innerError);

		differences = factory.makeCache(rowSize, columnSize);
	}

	@Override
	public void doForward() {
		MathMatrix outputData = outputKeyValue.getKey();
		MathMatrix leftInputData = inputKeyValues[0].getKey();
		MathMatrix rightInputData = inputKeyValues[1].getKey();
		for (int rowIndex = 0, rowSize = outputData.getRowSize(); rowIndex < rowSize; rowIndex++) {
			float value = 0F;
			for (int columnIndex = 0; columnIndex < columnSize; columnIndex++) {
				float difference = leftInputData.getValue(rowIndex, columnIndex) - rightInputData.getValue(rowIndex, columnIndex);
				// 缓存
				differences.setValue(rowIndex, columnIndex, difference);
				value += FastMath.pow(difference, 2);
			}
			outputData.setValue(rowIndex, 0, (float) FastMath.sqrt(value));
		}
		MathMatrix innerError = outputKeyValue.getValue();
		innerError.mapValues(MatrixMapper.ZERO, null, MathCalculator.SERIAL);
	}

	@Override
	public void doBackward() {
		MathMatrix outputData = outputKeyValue.getKey();
		MathMatrix innerError = outputKeyValue.getValue();

		MathMatrix leftInputError = inputKeyValues[0].getValue();
		MathMatrix rightInputError = inputKeyValues[1].getValue();
		for (int rowIndex = 0, rowSize = innerError.getRowSize(); rowIndex < rowSize; rowIndex++) {
			// innerError / outputData
			float error = outputData.getValue(rowIndex, 0);
			error = error < epsilon ? epsilon : error;
			error = innerError.getValue(rowIndex, 0) / error;
			for (int columnIndex = 0; columnIndex < columnSize; columnIndex++) {
				float value = differences.getValue(rowIndex, columnIndex) * error;
				if (leftInputError != null) {
					// TODO 使用累计的方式计算
					// TODO 需要锁机制,否则并发计算会导致Bug
					leftInputError.shiftValue(rowIndex, columnIndex, value);
				}
				if (rightInputError != null) {
					// TODO 使用累计的方式计算
					// TODO 需要锁机制,否则并发计算会导致Bug
					rightInputError.shiftValue(rowIndex, columnIndex, -value);
				}
			}
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
			EuclideanVertex that = (EuclideanVertex) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.vertexName, that.vertexName);
			equal.append(this.epsilon, that.epsilon);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vertexName);
		hash.append(epsilon);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "EuclideanVertex(name=" + vertexName + ")";
	}

}
