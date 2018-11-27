package com.jstarcraft.module.math.structure.matrix;

import java.util.concurrent.Future;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.message.SumMessage;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.MathVector;

public class ColumnArrayMatrixTestCase extends MatrixTestCase {

	@Override
	protected ColumnArrayMatrix getRandomMatrix(int dimension) {

		Table<Integer, Integer, Float> table = HashBasedTable.create();
		for (int rowIndex = 0; rowIndex < dimension; rowIndex++) {
			for (int columnIndex = 0; columnIndex < dimension; columnIndex++) {
				if (RandomUtility.randomBoolean()) {
					table.put(rowIndex, columnIndex, 0F);
				}
			}
		}
		SparseMatrix data = SparseMatrix.valueOf(dimension, dimension, table);
		ArrayVector[] vectors = new ArrayVector[dimension];
		for (int columnIndex = 0; columnIndex < dimension; columnIndex++) {
			vectors[columnIndex] = new ArrayVector(data.getColumnVector(columnIndex), (index, value, message) -> {
				return value;
			});
		}
		ColumnArrayMatrix matrix = ColumnArrayMatrix.valueOf(dimension, vectors);
		matrix.mapValues(MatrixMapper.randomOf(dimension), null, MathCalculator.SERIAL);
		return matrix;
	}

	@Override
	protected ColumnArrayMatrix getZeroMatrix(int dimension) {
		Table<Integer, Integer, Float> table = HashBasedTable.create();
		for (int rowIndex = 0; rowIndex < dimension; rowIndex++) {
			for (int columnIndex = 0; columnIndex < dimension; columnIndex++) {
				table.put(rowIndex, columnIndex, 0F);
			}
		}
		SparseMatrix data = SparseMatrix.valueOf(dimension, dimension, table);
		ArrayVector[] vectors = new ArrayVector[dimension];
		for (int columnIndex = 0; columnIndex < dimension; columnIndex++) {
			vectors[columnIndex] = new ArrayVector(data.getColumnVector(columnIndex), (index, value, message) -> {
				return value;
			});
		}
		ColumnArrayMatrix matrix = ColumnArrayMatrix.valueOf(dimension, vectors);
		return matrix;
	}

	@Override
	public void testProduct() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int dimension = 10;
			MathMatrix leftMatrix = getRandomMatrix(dimension);
			MathMatrix rightMatrix = getRandomMatrix(dimension);
			MathMatrix dataMatrix = getZeroMatrix(dimension);
			MathMatrix labelMatrix = DenseMatrix.valueOf(dimension, dimension);
			MathVector dataVector = dataMatrix.getColumnVector(0);
			MathVector labelVector = labelMatrix.getColumnVector(0);

			// 相当于transposeProductThis
			labelMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			// 相当于transposeProductThat
			labelMatrix.dotProduct(leftMatrix, true, rightMatrix, false, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftMatrix, true, rightMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftMatrix, true, rightMatrix, false, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			MathVector leftVector = leftMatrix.getColumnVector(RandomUtility.randomInteger(dimension));
			MathVector rightVector = rightMatrix.getColumnVector(RandomUtility.randomInteger(dimension));
			labelMatrix.dotProduct(leftVector, rightVector, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftVector, rightVector, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftVector, rightVector, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			labelVector.dotProduct(leftMatrix, true, rightVector, MathCalculator.SERIAL);
			dataVector.dotProduct(leftMatrix, true, rightVector, MathCalculator.SERIAL);
			Assert.assertTrue(equalVector(dataVector, labelVector));
			dataVector.dotProduct(leftMatrix, true, rightVector, MathCalculator.PARALLEL);
			Assert.assertTrue(equalVector(dataVector, labelVector));

			labelVector.dotProduct(leftVector, rightMatrix, false, MathCalculator.SERIAL);
			dataVector.dotProduct(leftVector, rightMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalVector(dataVector, labelVector));
			dataVector.dotProduct(leftVector, rightMatrix, false, MathCalculator.PARALLEL);
			Assert.assertTrue(equalVector(dataVector, labelVector));

			// 利用转置乘运算的对称性
			dataMatrix = new SymmetryMatrix(dimension);
			labelMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
		});
		task.get();
	}

	@Test
	public void testNotify() {
		int dimension = 10;
		ColumnArrayMatrix matrix = getRandomMatrix(dimension);
		matrix.mapValues(MatrixMapper.constantOf(1F), null, MathCalculator.SERIAL);

		try {
			matrix.getRowVector(RandomUtility.randomInteger(dimension));
			Assert.fail();
		} catch (UnsupportedOperationException exception) {
		}

		ArrayVector vector = matrix.getColumnVector(RandomUtility.randomInteger(dimension));
		int oldSize = vector.getElementSize();
		int newSize = RandomUtility.randomInteger(oldSize);
		int[] indexes = new int[newSize];
		for (int index = 0; index < newSize; index++) {
			indexes[index] = index;
		}
		SumMessage message = new SumMessage(false);
		matrix.attachMonitor((iterator, oldElementSize, newElementSize, oldKnownSize, newKnownSize, oldUnknownSize, newUnknownSize) -> {
			Assert.assertThat(newElementSize - oldElementSize, CoreMatchers.equalTo(newSize - oldSize));
			message.accumulateValue(oldSize + newSize);
		});
		vector.modifyIndexes((index, value, information) -> {
			return 1F;
		}, null, indexes);
		Assert.assertThat(message.getValue(), CoreMatchers.equalTo(oldSize + newSize + 0F));
		Assert.assertThat(matrix.getSum(false), CoreMatchers.equalTo(matrix.getElementSize() + 0F));

		message.accumulateValue(-message.getValue());
		matrix.collectValues((row, column, value, information) -> {
			message.accumulateValue(value);
		}, message, MathCalculator.SERIAL);
		Assert.assertThat(message.getValue(), CoreMatchers.equalTo(matrix.getSum(false)));

		message.accumulateValue(-message.getValue());
		for (MatrixScalar term : matrix) {
			message.accumulateValue(term.getValue());
		}
		Assert.assertThat(message.getValue(), CoreMatchers.equalTo(matrix.getSum(false)));
	}

}
