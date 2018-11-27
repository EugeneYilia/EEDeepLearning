package com.jstarcraft.module.math.structure.matrix;

import com.jstarcraft.module.math.structure.MathCalculator;

public class DenseMatrixTestCase extends MatrixTestCase {

	@Override
	protected DenseMatrix getRandomMatrix(int dimension) {
		DenseMatrix matrix = DenseMatrix.valueOf(dimension, dimension);
		matrix.mapValues(MatrixMapper.randomOf(dimension), null, MathCalculator.SERIAL);
		return matrix;
	}

	@Override
	protected DenseMatrix getZeroMatrix(int dimension) {
		DenseMatrix matrix = DenseMatrix.valueOf(dimension, dimension);
		return matrix;
	}

}
