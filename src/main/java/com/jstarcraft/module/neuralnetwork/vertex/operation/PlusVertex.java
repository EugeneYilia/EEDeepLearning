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
import com.jstarcraft.module.math.structure.matrix.ColumnCompositeMatrix;
import com.jstarcraft.module.math.structure.matrix.CompositeMatrix;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.RowCompositeMatrix;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.AbstractVertex;

/**
 * An ElementWiseVertex is used to combine the activations of two or more layer
 * in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or
 * multiplication or by selecting the maximum. Addition, Average, Product and
 * Max may use an arbitrary number of input arrays. Note that in the case of
 * subtraction, only two inputs may be used. In all cases, the shape of the
 * input arrays must be identical.
 * 
 * @author Alex Black
 */
// TODO 准备改名为AddVertex
public class PlusVertex extends AbstractVertex {

	protected PlusVertex() {
	}

	public PlusVertex(String name, MatrixFactory factory) {
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

		// 检查样本的维度是否一样
		int columnSize = samples[0].getKey().getColumnSize();
		for (int position = 1; position < samples.length; position++) {
			if (columnSize != samples[position].getKey().getColumnSize()) {
				throw new IllegalArgumentException();
			}
		}

		// TODO 考虑支持CompositeMatrix.
		if (samples[0].getKey() instanceof CompositeMatrix) {
			CompositeMatrix matrix = CompositeMatrix.class.cast(samples[0].getKey());
			int size = matrix.getComponentSize();
			MathMatrix[] outputDatas = new MathMatrix[size];
			MathMatrix[] innerErrors = new MathMatrix[size];
			for (int index = 0; index < size; index++) {
				MathMatrix component = matrix.getComponentMatrix(index);
				outputDatas[index] = factory.makeCache(component.getRowSize(), component.getColumnSize());
				innerErrors[index] = factory.makeCache(component.getRowSize(), component.getColumnSize());
			}
			if (samples[0].getKey() instanceof ColumnCompositeMatrix) {
				MathMatrix outputData = ColumnCompositeMatrix.attachOf(outputDatas);
				outputKeyValue.setKey(outputData);
				MathMatrix innerError = ColumnCompositeMatrix.attachOf(innerErrors);
				outputKeyValue.setValue(innerError);
			}
			if (samples[0].getKey() instanceof RowCompositeMatrix) {
				MathMatrix outputData = RowCompositeMatrix.attachOf(outputDatas);
				outputKeyValue.setKey(outputData);
				MathMatrix innerError = RowCompositeMatrix.attachOf(innerErrors);
				outputKeyValue.setValue(innerError);
			}
		} else {
			MathMatrix outputData = factory.makeCache(rowSize, columnSize);
			outputKeyValue.setKey(outputData);
			MathMatrix innerError = factory.makeCache(rowSize, columnSize);
			outputKeyValue.setValue(innerError);
		}
	}

	@Override
	public void doForward() {
		MathMatrix outputData = outputKeyValue.getKey();
		outputData.setValues(0F);
		for (KeyValue<MathMatrix, MathMatrix> keyValue : inputKeyValues) {
			MathMatrix inputData = keyValue.getKey();
			outputData.addMatrix(inputData, false);
		}
		MathMatrix innerError = outputKeyValue.getValue();
		innerError.setValues(0F);
	}

	@Override
	public void doBackward() {
		MathMatrix innerError = outputKeyValue.getValue();
		for (KeyValue<MathMatrix, MathMatrix> keyValue : inputKeyValues) {
			MathMatrix outerError = keyValue.getValue();
			if (outerError != null) {
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				synchronized (outerError) {
					outerError.addMatrix(innerError, false);
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
			PlusVertex that = (PlusVertex) object;
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
		return "PlusVertex(name=" + vertexName + ")";
	}

}
