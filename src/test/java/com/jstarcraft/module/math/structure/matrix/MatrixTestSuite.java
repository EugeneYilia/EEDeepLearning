package com.jstarcraft.module.math.structure.matrix;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({

		ColumnArrayMatrixTestCase.class,

		ColumnCompositeMatrixTestCase.class,

		DenseMatrixTestCase.class,

		Nd4jMatrixTestCase.class,
		
		RowArrayMatrixTestCase.class,

		RowCompositeMatrixTestCase.class,

		SparseMatrixTestCase.class,

		SymmetryMatrixTestCase.class })
public class MatrixTestSuite {

}
