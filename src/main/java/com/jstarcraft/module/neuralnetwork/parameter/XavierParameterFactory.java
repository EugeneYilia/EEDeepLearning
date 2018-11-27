package com.jstarcraft.module.neuralnetwork.parameter;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

public class XavierParameterFactory implements ParameterFactory {

	@Override
	public void setValues(MathMatrix matrix) {
		INDArray ndArray = Nd4j.randn(new int[] { matrix.getRowSize(), matrix.getColumnSize() }).muli(FastMath.sqrt(2D / (matrix.getRowSize() + matrix.getColumnSize())));
		matrix.mapValues((row, column, value, message) -> {
			return ndArray.getFloat(row, column);
		}, null, MathCalculator.SERIAL);
	}

}
