package com.jstarcraft.module.neuralnetwork.vertex;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.CompositeMatrix;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.transformation.HorizontalStackVertex;

@ModelDefinition(value = { "vertexName", "factory", "orientation" })
// TODO 准备整合到StackVertex
@Deprecated
public class Nd4jVertex extends AbstractVertex {

	protected boolean orientation;

	protected Nd4jVertex() {
	}

	public Nd4jVertex(String name, MatrixFactory factory, boolean orientation) {
		super(name, factory);
		this.orientation = orientation;
	}

	@Override
	public void doCache(KeyValue<MathMatrix, MathMatrix>... samples) {
		super.doCache(samples);

		// 检查样本
		if (samples.length != 1) {
			throw new IllegalArgumentException();
		}

		CompositeMatrix keyMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getKey());
		MathMatrix outputData = new Nd4jMatrix(Nd4j.zeros(keyMatrix.getRowSize(), keyMatrix.getColumnSize()));
		outputKeyValue.setKey(outputData);

		CompositeMatrix valueMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getValue());
		MathMatrix innerError = new Nd4jMatrix(Nd4j.zeros(valueMatrix.getRowSize(), valueMatrix.getColumnSize()));
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
		CompositeMatrix inputKeyMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getKey());
		CompositeMatrix inputValueMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getValue());
		Nd4jMatrix outputKeyMatrix = Nd4jMatrix.class.cast(outputKeyValue.getKey());
		Nd4jMatrix outputValueMatrix = Nd4jMatrix.class.cast(outputKeyValue.getValue());

		{
			INDArray outputData = outputKeyMatrix.getArray();
			int cursor = 0;
			for (MathMatrix component : inputKeyMatrix.getComponentMatrixes()) {
				Nd4jMatrix nd4j = Nd4jMatrix.class.cast(component);
				INDArray array = nd4j.getArray();
				if (orientation) {
					outputData.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(cursor, cursor + array.columns()) }, array);
					cursor += array.columns();
				} else {
					outputData.put(new INDArrayIndex[] { NDArrayIndex.interval(cursor, cursor + array.rows()), NDArrayIndex.all() }, array);
					cursor += array.rows();
				}
			}
		}

		outputValueMatrix.setValues(0F);
	}

	@Override
	public void doBackward() {
		CompositeMatrix inputKeyMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getKey());
		CompositeMatrix inputValueMatrix = CompositeMatrix.class.cast(inputKeyValues[0].getValue());
		Nd4jMatrix outputKeyMatrix = Nd4jMatrix.class.cast(outputKeyValue.getKey());
		Nd4jMatrix outputValueMatrix = Nd4jMatrix.class.cast(outputKeyValue.getValue());

		{
			INDArray innerError = outputValueMatrix.getArray();
			int cursor = 0;
			for (MathMatrix component : inputValueMatrix.getComponentMatrixes()) {
				// TODO 使用累计的方式计算
				// TODO 需要锁机制,否则并发计算会导致Bug
				Nd4jMatrix nd4j = Nd4jMatrix.class.cast(component);
				INDArray array = nd4j.getArray();
				synchronized (component) {
					if (orientation) {
						array.addi(innerError.get(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(cursor, cursor + array.columns()) }));
						cursor += array.columns();
					} else {
						array.addi(innerError.get(new INDArrayIndex[] { NDArrayIndex.interval(cursor, cursor + array.rows()), NDArrayIndex.all() }));
						cursor += array.rows();
					}
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
			HorizontalStackVertex that = (HorizontalStackVertex) object;
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
		return "Nd4jVertex(name=" + vertexName + ")";
	}

}
