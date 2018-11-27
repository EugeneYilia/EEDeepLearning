package com.jstarcraft.module.math.algorithm.distribution;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.algorithm.distribution.Probability;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;

public abstract class ProbabilityTestCase {

	protected static DenseMatrix getMatrix(INDArray array) {
		return DenseMatrix.valueOf(array.rows(), array.columns(), (row, column, value, message) -> {
			return array.getFloat(row, column);
		});
	}

	private static boolean equalMatrix(DenseMatrix matrix, INDArray array) {
		for (int row = 0; row < matrix.getRowSize(); row++) {
			for (int column = 0; column < matrix.getColumnSize(); column++) {
				if (!MathUtility.equal(matrix.getValue(row, column), array.getFloat(row, column))) {
					return false;
				}
			}
		}
		return true;
	}

	protected abstract void assertSample(Probability<Number> newFuction, Distribution oldFunction);

	protected abstract Distribution getOldFunction(long seed);

	protected abstract <T extends Number> Probability<T> getNewFunction(long seed);

	@Test
	public void test() {
		long seed = 100L;

		Distribution oldFunction = getOldFunction(seed);
		Probability<Number> newFuction = getNewFunction(seed);

		for (int index = 0; index < seed; index++) {
			newFuction.setSeed(index);
			oldFunction.reseedRandomGenerator(index);
			assertSample(newFuction, oldFunction);
		}

		Assert.assertThat(newFuction.getMaximum().doubleValue(), CoreMatchers.equalTo(oldFunction.getSupportUpperBound()));
		Assert.assertThat(newFuction.getMinimum().doubleValue(), CoreMatchers.equalTo(oldFunction.getSupportLowerBound()));
		Assert.assertThat(newFuction.inverseDistribution(1D).doubleValue(), CoreMatchers.equalTo(oldFunction.getSupportUpperBound()));
		Assert.assertThat(newFuction.inverseDistribution(0D).doubleValue(), CoreMatchers.equalTo(oldFunction.getSupportLowerBound()));
		Assert.assertThat(newFuction.cumulativeDistribution(newFuction.getMaximum()), CoreMatchers.equalTo(oldFunction.cumulativeProbability(oldFunction.getSupportUpperBound())));
		Assert.assertThat(newFuction.cumulativeDistribution(newFuction.getMinimum()), CoreMatchers.equalTo(oldFunction.cumulativeProbability(oldFunction.getSupportLowerBound())));
	}

}
