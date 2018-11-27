package com.jstarcraft.module.math.structure.matrix;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.math.structure.MathCalculator;

public class Nd4jMatrixTestCase extends MatrixTestCase {

	@Override
	protected Nd4jMatrix getRandomMatrix(int dimension) {
		INDArray array = Nd4j.zeros(dimension, dimension, 'c');
		Nd4jMatrix matrix = new Nd4jMatrix(array);
		matrix.mapValues(MatrixMapper.randomOf(dimension), null, MathCalculator.SERIAL);
		return matrix;
	}

	@Override
	protected Nd4jMatrix getZeroMatrix(int dimension) {
		INDArray array = Nd4j.zeros(dimension, dimension, 'f');
		Nd4jMatrix matrix = new Nd4jMatrix(array);
		return matrix;
	}

}
