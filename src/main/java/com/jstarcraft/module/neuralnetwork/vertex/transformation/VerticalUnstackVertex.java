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

package com.jstarcraft.module.neuralnetwork.vertex.transformation;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.RowCompositeMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.AbstractVertex;

/**
 * UnstackVertex allows for unstacking of inputs so that they may be forwarded
 * through a network. This is useful for cases such as Triplet Embedding, where
 * embeddings can be separated and run through subsequent layers.
 *
 * Works similarly to SubsetVertex, except on dimension 0 of the input.
 * stackSize is explicitly defined by the user to properly calculate an step.
 *
 * @author Justin Long (crockpotveggies)
 */
@ModelDefinition(value = { "vertexName", "factory", "from", "to" })
// TODO 准备改名为VerticalDetachVertex
public class VerticalUnstackVertex extends AbstractVertex {

	// inclusive
	private int from;

	// exclusive
	private int to;

	protected VerticalUnstackVertex() {
	}

	public VerticalUnstackVertex(String name, MatrixFactory factory, int from, int to) {
		super(name, factory);
		this.from = from;
		this.to = to;
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		super.doCache(samples);

		// 获取样本的数量与维度
		MathMatrix outputData = RowCompositeMatrix.detachOf(RowCompositeMatrix.class.cast(samples[0].getKey()), from, to);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = RowCompositeMatrix.detachOf(RowCompositeMatrix.class.cast(samples[0].getValue()), from, to);
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
	}

	@Override
	public void doBackward() {
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
			VerticalUnstackVertex that = (VerticalUnstackVertex) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.vertexName, that.vertexName);
			equal.append(this.from, that.from);
			equal.append(this.to, that.to);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vertexName);
		hash.append(from);
		hash.append(to);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "UnstackVertex(name=" + vertexName + ", from=" + from + ", to=" + to + ")";
	}

}
