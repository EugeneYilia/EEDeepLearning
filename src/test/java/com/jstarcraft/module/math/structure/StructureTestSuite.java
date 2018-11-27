package com.jstarcraft.module.math.structure;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

import com.jstarcraft.module.math.structure.matrix.MatrixTestSuite;
import com.jstarcraft.module.math.structure.tensor.SparseTensorTestCase;

@RunWith(Suite.class)
@SuiteClasses({

		MathIteratorTestCase.class,

		MatrixTestSuite.class,

		SparseTensorTestCase.class })
public class StructureTestSuite {

}
