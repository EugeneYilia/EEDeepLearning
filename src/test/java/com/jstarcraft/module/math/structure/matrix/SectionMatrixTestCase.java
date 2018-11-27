package com.jstarcraft.module.math.structure.matrix;

import com.jstarcraft.module.math.structure.MathCalculator;

public class SectionMatrixTestCase extends MatrixTestCase {

	@Override
	protected SectionMatrix getRandomMatrix(int dimension) {
		SectionMatrix matrix = new SectionMatrix(DenseMatrix.valueOf(dimension * 3, dimension * 3), dimension, dimension * 2, dimension, dimension * 2);
		matrix.mapValues(MatrixMapper.randomOf(dimension), null, MathCalculator.SERIAL);
		return matrix;
	}

	@Override
	protected SectionMatrix getZeroMatrix(int dimension) {
		SectionMatrix matrix = new SectionMatrix(DenseMatrix.valueOf(dimension * 3, dimension * 3), dimension, dimension * 2, dimension, dimension * 2);
		return matrix;
	}

}
