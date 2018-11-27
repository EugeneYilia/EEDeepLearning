package com.jstarcraft.module.neuralnetwork;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;

public class DenseMatrixFactory implements MatrixFactory {

	@Override
	public MathMatrix makeCache(int rowSize, int columnSize) {
		DenseMatrix matrix = DenseMatrix.valueOf(rowSize, columnSize);
		return matrix;
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
		return "DenseMatrixFactory()";
	}

}
