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

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathScalar;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.AbstractVertex;

/**
 * 按行向量,两两做外积运算
 * 
 * @author Birdy
 *
 */
public class OuterProductVertex extends AbstractVertex {

	private int rowSize;

	private int columnSize;

	protected OuterProductVertex() {
	}

	public OuterProductVertex(String name, MatrixFactory factory) {
		super(name, factory);
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		if (samples.length != 2) {
			throw new IllegalArgumentException();
		}

		super.doCache(samples);

		// 检查样本的数量是否一样
		rowSize = samples[0].getKey().getRowSize();
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

		MathMatrix outputData = factory.makeCache(rowSize, columnSize * columnSize);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = factory.makeCache(rowSize, columnSize * columnSize);
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
		MathMatrix outputData = outputKeyValue.getKey();
		MathMatrix leftInputData = inputKeyValues[0].getKey();
		MathMatrix rightInputData = inputKeyValues[1].getKey();
		for (int rowIndex = 0, rowSize = outputData.getRowSize(); rowIndex < rowSize; rowIndex++) {
			MathVector leftVector = leftInputData.getRowVector(rowIndex), rightVector = rightInputData.getRowVector(rowIndex);
			outputData.getRowVector(rowIndex).mapValues((index, value, message) -> {
				int leftIndex = index / columnSize;
				int rightIndex = index % columnSize;
				return leftVector.getValue(leftIndex) * rightVector.getValue(rightIndex);
			}, null, MathCalculator.SERIAL);
		}
		MathMatrix innerError = outputKeyValue.getValue();
		innerError.mapValues(MatrixMapper.ZERO, null, MathCalculator.SERIAL);
	}

	@Override
	public void doBackward() {
		MathMatrix leftInputData = inputKeyValues[0].getKey();
		MathMatrix rightInputData = inputKeyValues[1].getKey();
		MathMatrix innerError = outputKeyValue.getValue();

		MathMatrix leftInputError = inputKeyValues[0].getValue();
		MathMatrix rightInputError = inputKeyValues[1].getValue();
		MathMatrix errorMatrix = factory.makeCache(columnSize, columnSize);
		for (int rowIndex = 0, rowSize = innerError.getRowSize(); rowIndex < rowSize; rowIndex++) {
			MathVector errorVector = innerError.getRowVector(rowIndex);
			errorMatrix.mapValues((row, column, value, message) -> {
				return errorVector.getValue(row * columnSize + column);
			}, null, MathCalculator.SERIAL);
			if (leftInputError != null) {
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				synchronized (leftInputError) {
					MathVector vector = rightInputData.getRowVector(rowIndex);
					leftInputError.getRowVector(rowIndex).accumulateProduct(errorMatrix, false, vector, MathCalculator.SERIAL);
				}
			}
			if (rightInputError != null) {
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				synchronized (rightInputError) {
					MathVector vector = leftInputData.getRowVector(rowIndex);
					rightInputError.getRowVector(rowIndex).accumulateProduct(vector, errorMatrix, false, MathCalculator.SERIAL);
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
			OuterProductVertex that = (OuterProductVertex) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.vertexName, that.vertexName);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vertexName);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "ProductVertex(name=" + vertexName + ")";
	}

}
