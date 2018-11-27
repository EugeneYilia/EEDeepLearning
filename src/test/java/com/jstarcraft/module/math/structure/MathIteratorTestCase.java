package com.jstarcraft.module.math.structure;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.Stream;

import org.junit.Assert;
import org.junit.Test;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixCollector;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.message.BoundaryMessage;
import com.jstarcraft.module.math.structure.message.NormMessage;
import com.jstarcraft.module.math.structure.message.SumMessage;
import com.jstarcraft.module.math.structure.message.VarianceMessage;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;

public class MathIteratorTestCase {

	@Test
	public void testBoundary() {
		MathMatrix matrix = DenseMatrix.valueOf(5, 10);
		matrix.mapValues(MatrixMapper.RANDOM, null, MathCalculator.SERIAL);

		{
			BoundaryMessage message = new BoundaryMessage(false);
			matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);

			KeyValue<Float, Float> keyValue = matrix.getBoundary(false);

			Assert.assertEquals("最小值比较", message.getValue().getKey(), keyValue.getKey(), MathUtility.EPSILON);
			Assert.assertEquals("最大值比较", message.getValue().getValue(), keyValue.getValue(), MathUtility.EPSILON);
		}

		{
			BoundaryMessage message = new BoundaryMessage(true);
			matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);

			KeyValue<Float, Float> keyValue = matrix.getBoundary(true);

			Assert.assertEquals("最小值比较", message.getValue().getKey(), keyValue.getKey(), MathUtility.EPSILON);
			Assert.assertEquals("最大值比较", message.getValue().getValue(), keyValue.getValue(), MathUtility.EPSILON);
		}
	}

	private double getMedian(ArrayList<Integer> data) {
		int lenght = data.size();
		// 中位数
		double median = 0;
		Collections.sort(data);
		if (lenght % 2 == 0)
			median = (data.get((lenght - 1) / 2) + data.get(lenght / 2)) / 2D;
		else
			median = data.get(lenght / 2);
		return median;
	}

	@Test
	public void testMedian() {
		Stream.iterate(1, size -> size + 1).limit(100).forEach(size -> {
			MathVector vector = DenseVector.valueOf(size);
			ArrayList<Integer> data = new ArrayList<>(size);

			Stream.iterate(0, times -> times).limit(100).forEach(times -> {
				for (int index = 0; index < size; index++) {
					int value = RandomUtility.randomInteger(-10, 10);
					vector.setValue(index, value);
					data.add(value);
				}
				Assert.assertEquals("中位数值比较", getMedian(data), vector.getMedian(false), MathUtility.EPSILON);
				data.clear();
			});
		});
	}

	@Test
	public void testNorm() {
		MathMatrix matrix = DenseMatrix.valueOf(5, 10);
		matrix.mapValues(MatrixMapper.RANDOM, null, MathCalculator.SERIAL);

		for (int index = 0; index < 10; index++) {
			NormMessage message = new NormMessage(index);
			matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);

			if (index == 0) {
				Assert.assertEquals("范数值比较", message.getValue(), matrix.getNorm(index), MathUtility.EPSILON);
			} else {
				Assert.assertEquals("范数值比较", Math.pow(message.getValue(), 1D / message.getPower()), matrix.getNorm(index), MathUtility.EPSILON);
			}
		}
	}

	@Test
	public void testVariance() {
		MathMatrix matrix = DenseMatrix.valueOf(5, 10);
		matrix.mapValues(MatrixMapper.RANDOM, null, MathCalculator.SERIAL);

		float mean = matrix.getSum(false) / matrix.getElementSize();
		VarianceMessage message = new VarianceMessage(mean);
		matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);

		KeyValue<Float, Float> keyValue = matrix.getVariance();

		Assert.assertEquals("平均值比较", message.getMean(), keyValue.getKey(), MathUtility.EPSILON);
		Assert.assertEquals("方差值比较", message.getValue() / matrix.getElementSize(), keyValue.getValue(), MathUtility.EPSILON);
	}

	@Test
	public void testSum() {
		MathMatrix matrix = DenseMatrix.valueOf(5, 10);
		matrix.mapValues(MatrixMapper.RANDOM, null, MathCalculator.SERIAL);

		{
			SumMessage message = new SumMessage(false);
			matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);
			Assert.assertEquals("总值比较", message.getValue(), matrix.getSum(false), MathUtility.EPSILON);
		}

		{
			SumMessage message = new SumMessage(true);
			matrix.collectValues(MatrixCollector.ACCUMULATOR, message, MathCalculator.SERIAL);
			Assert.assertEquals("总值比较", message.getValue(), matrix.getSum(true), MathUtility.EPSILON);
		}
	}

}
