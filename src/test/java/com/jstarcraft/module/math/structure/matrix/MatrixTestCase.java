package com.jstarcraft.module.math.structure.matrix;

import java.util.concurrent.Future;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MockMessage;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.model.ModelCodec;

public abstract class MatrixTestCase {

	private MatrixCollector<MockMessage> collector = new MatrixCollector<MockMessage>() {

		@Override
		public void collect(int row, int column, float value, MockMessage message) {
			try {
				Thread.sleep(0L);
				message.accumulateValue(value);
			} catch (Exception exception) {
			}
		}

	};

	private MatrixMapper<MockMessage> mapper = new MatrixMapper<MockMessage>() {

		@Override
		public float map(int row, int column, float value, MockMessage message) {
			try {
				Thread.sleep(0L);
				value += 1D;
				message.accumulateValue(value);
			} catch (Exception exception) {
			}
			return value;
		}

	};

	protected abstract MathMatrix getRandomMatrix(int dimension);

	protected abstract MathMatrix getZeroMatrix(int dimension);

	protected static boolean equalMatrix(MathMatrix left, MathMatrix right) {
		for (MatrixScalar term : left) {
			if (!MathUtility.equal(term.getValue(), right.getValue(term.getRow(), term.getColumn()))) {
				System.out.println(term.getValue());
				System.out.println(right.getValue(term.getRow(), term.getColumn()));
				return false;
			}
		}
		return true;
	}

	protected static boolean equalVector(MathVector left, MathVector right) {
		for (VectorScalar term : left) {
			if (!MathUtility.equal(term.getValue(), right.getValue(term.getIndex()))) {
				System.out.println(term.getValue());
				System.out.println(right.getValue(term.getIndex()));
				return false;
			}
		}
		return true;
	}

	@Test
	public void testCalculate() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int dimension = 10;
			MathMatrix dataMatrix = getRandomMatrix(dimension);

			MockMessage oldMessage = new MockMessage();
			MockMessage newMessage = new MockMessage();
			dataMatrix.collectValues(collector, oldMessage, MathCalculator.SERIAL);
			Assert.assertThat(oldMessage.getValue(), CoreMatchers.equalTo(dataMatrix.getSum(false)));
			dataMatrix.mapValues(mapper, newMessage, MathCalculator.SERIAL);
			Assert.assertThat(newMessage.getValue(), CoreMatchers.equalTo(dataMatrix.getSum(false)));
			Assert.assertTrue(dataMatrix.getElementSize() == (newMessage.getValue() - oldMessage.getValue()));
			Assert.assertThat(oldMessage.getAttachTimes(), CoreMatchers.equalTo(0));
			Assert.assertThat(oldMessage.getDetachTimes(), CoreMatchers.equalTo(0));
			Assert.assertThat(newMessage.getAttachTimes(), CoreMatchers.equalTo(0));
			Assert.assertThat(newMessage.getDetachTimes(), CoreMatchers.equalTo(0));

			oldMessage = new MockMessage();
			newMessage = new MockMessage();
			dataMatrix.collectValues(collector, oldMessage, MathCalculator.PARALLEL);
			Assert.assertThat(oldMessage.getValue(), CoreMatchers.equalTo(dataMatrix.getSum(false)));
			dataMatrix.mapValues(mapper, newMessage, MathCalculator.PARALLEL);
			Assert.assertThat(newMessage.getValue(), CoreMatchers.equalTo(dataMatrix.getSum(false)));
			Assert.assertTrue(dataMatrix.getElementSize() == (newMessage.getValue() - oldMessage.getValue()));
			Assert.assertTrue(oldMessage.getAttachTimes() > 0);
			Assert.assertTrue(oldMessage.getDetachTimes() > 0);
			Assert.assertTrue(newMessage.getAttachTimes() > 0);
			Assert.assertTrue(newMessage.getDetachTimes() > 0);
		});
		task.get();
	}

	@Test
	public void testProduct() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int dimension = 10;
			MathMatrix leftMatrix = getRandomMatrix(dimension);
			MathMatrix rightMatrix = getRandomMatrix(dimension);
			MathMatrix dataMatrix = getZeroMatrix(dimension);
			MathMatrix labelMatrix = DenseMatrix.valueOf(dimension, dimension);
			MathVector dataVector = dataMatrix.getRowVector(0);
			MathVector labelVector = labelMatrix.getRowVector(0);

			// 相当于transposeProductThis
			labelMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftMatrix, true, leftMatrix, false, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			// 相当于transposeProductThat
			labelMatrix.dotProduct(leftMatrix, false, rightMatrix, false, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftMatrix, false, rightMatrix, false, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftMatrix, false, rightMatrix, false, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			MathVector leftVector = leftMatrix.getRowVector(RandomUtility.randomInteger(dimension));
			MathVector rightVector = rightMatrix.getColumnVector(RandomUtility.randomInteger(dimension));
			labelMatrix.dotProduct(leftVector, rightVector, MathCalculator.SERIAL);
			dataMatrix.dotProduct(leftVector, rightVector, MathCalculator.SERIAL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));
			dataMatrix.dotProduct(leftVector, rightVector, MathCalculator.PARALLEL);
			Assert.assertTrue(equalMatrix(dataMatrix, labelMatrix));

			labelVector.dotProduct(leftMatrix, false, rightVector, MathCalculator.SERIAL);
			dataVector.dotProduct(leftMatrix, false, rightVector, MathCalculator.SERIAL);
			Assert.assertTrue(equalVector(dataVector, labelVector));
			dataVector.dotProduct(leftMatrix, false, rightVector, MathCalculator.PARALLEL);
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
	public void testSize() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int dimension = 10;
			MathMatrix dataMatrix = getRandomMatrix(dimension);

			Assert.assertThat(dataMatrix.getKnownSize() + dataMatrix.getUnknownSize(), CoreMatchers.equalTo(dataMatrix.getRowSize() * dataMatrix.getColumnSize()));

			int elementSize = 0;
			float sumValue = 0F;
			for (MatrixScalar term : dataMatrix) {
				elementSize++;
				sumValue += term.getValue();
			}
			Assert.assertThat(elementSize, CoreMatchers.equalTo(dataMatrix.getElementSize()));
			Assert.assertThat(sumValue, CoreMatchers.equalTo(dataMatrix.getSum(false)));
		});
		task.get();
	}

	@Test
	public void testSum() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int dimension = 10;
			MathMatrix dataMatrix = getRandomMatrix(dimension);

			float oldSum = dataMatrix.getSum(false);
			dataMatrix.scaleValues(2F);
			float newSum = dataMatrix.getSum(false);
			Assert.assertThat(newSum, CoreMatchers.equalTo(oldSum * 2F));

			oldSum = newSum;
			dataMatrix.shiftValues(1F);
			newSum = dataMatrix.getSum(false);
			Assert.assertThat(newSum, CoreMatchers.equalTo(oldSum + dataMatrix.getElementSize()));

			dataMatrix.setValues(0F);
			newSum = dataMatrix.getSum(false);
			Assert.assertThat(newSum, CoreMatchers.equalTo(0F));
		});
		task.get();
	}

	@Test
	public void testCodec() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			// 维度设置为100,可以测试编解码的效率.
			int dimension = 100;
			MathMatrix oldMatrix = getRandomMatrix(dimension);

			for (ModelCodec codec : ModelCodec.values()) {
				byte[] data = codec.encodeModel(oldMatrix);
				MathMatrix newMatrix = (MathMatrix) codec.decodeModel(data);
				Assert.assertThat(newMatrix, CoreMatchers.equalTo(oldMatrix));
			}
		});
		task.get();
	}

	@Test
	public void testFourArithmeticOperation() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			RandomUtility.setSeed(0L);
			int dimension = 10;
			MathMatrix dataMatrix = getZeroMatrix(dimension);
			dataMatrix.mapValues(MatrixMapper.randomOf(10F), null, MathCalculator.SERIAL);
			MathMatrix copyMatrix = getZeroMatrix(dimension);
			float sum = dataMatrix.getSum(false);

			copyMatrix.copyMatrix(dataMatrix, false);
			Assert.assertThat(copyMatrix.getSum(false), CoreMatchers.equalTo(sum));

			dataMatrix.subtractMatrix(copyMatrix, false);
			Assert.assertThat(dataMatrix.getSum(false), CoreMatchers.equalTo(0F));

			dataMatrix.addMatrix(copyMatrix, false);
			Assert.assertThat(dataMatrix.getSum(false), CoreMatchers.equalTo(sum));

			dataMatrix.divideMatrix(copyMatrix, false);
			Assert.assertThat(dataMatrix.getSum(false), CoreMatchers.equalTo(dataMatrix.getElementSize() + 0F));

			dataMatrix.multiplyMatrix(copyMatrix, false);
			Assert.assertThat(dataMatrix.getSum(false), CoreMatchers.equalTo(sum));
		});
		task.get();
	}

	@Test
	public void testPerformance() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {

		});
		task.get();
		// 性能测试
	}

}
