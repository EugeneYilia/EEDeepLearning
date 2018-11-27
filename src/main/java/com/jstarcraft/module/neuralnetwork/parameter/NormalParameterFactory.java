package com.jstarcraft.module.neuralnetwork.parameter;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

public class NormalParameterFactory implements ParameterFactory {

	@Override
	public void setValues(MathMatrix matrix) {
		INDArray ndArray = Nd4j.randn(new int[] { matrix.getRowSize(), matrix.getColumnSize() }).divi(FastMath.sqrt(matrix.getRowSize()));
		matrix.mapValues((row, column, value, message) -> {
			return ndArray.getFloat(row, column);
		}, null, MathCalculator.PARALLEL);
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
			return true;
		}
	}

	@Override
	public int hashCode() {
		return getClass().hashCode();
	}

	@Override
	public String toString() {
		return "NormalParameterFactory()";
	}

}
