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
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
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
public class SumVertex extends AbstractVertex {

	protected SumVertex() {
	}

	public SumVertex(String name, MatrixFactory factory) {
		super(name, factory);
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

		MathMatrix outputData = factory.makeCache(rowSize, 1);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = factory.makeCache(rowSize, 1);
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
		MathMatrix outputData = outputKeyValue.getKey();
		MathVector outputVector = outputData.getColumnVector(0);
		outputVector.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
		for (int index = 0; index < inputKeyValues.length; index++) {
			MathMatrix inputData = inputKeyValues[index].getKey();
			for (int columnIndex = 0, columnSize = inputData.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				MathVector inputVector = inputData.getColumnVector(columnIndex);
				outputVector.mapValues((position, value, message) -> {
					return value + inputVector.getValue(position);
				}, null, MathCalculator.SERIAL);
			}
		}
		MathMatrix innerError = outputKeyValue.getValue();
		innerError.mapValues(MatrixMapper.ZERO, null, MathCalculator.SERIAL);
	}

	@Override
	public void doBackward() {
		MathVector outputData = outputKeyValue.getKey().getColumnVector(0);
		MathVector innerError = outputKeyValue.getValue().getColumnVector(0);

		for (int index = 0; index < inputKeyValues.length; index++) {
			MathMatrix outerError = inputKeyValues[index].getValue();
			if (outerError != null) {
				MathMatrix inputData = inputKeyValues[index].getKey();
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				outerError.mapValues((row, column, value, message) -> {
					value += inputData.getValue(row, column) * innerError.getValue(row) / outputData.getValue(row);
					return value;
				}, null, MathCalculator.PARALLEL);
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
			SumVertex that = (SumVertex) object;
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
		return "SumVertex(name=" + vertexName + ")";
	}

}
