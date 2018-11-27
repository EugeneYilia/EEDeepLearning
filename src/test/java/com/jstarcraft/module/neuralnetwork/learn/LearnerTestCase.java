package com.jstarcraft.module.neuralnetwork.learn;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Future;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.model.ModelCodec;

public abstract class LearnerTestCase {

	private static DenseMatrix getMatrix(INDArray array) {
		return DenseMatrix.valueOf(array.rows(), array.columns(), (row, column, value, message) -> {
			return array.getFloat(row, column);
		});
	}

	private static boolean equalMatrix(MathMatrix matrix, INDArray array) {
		for (int row = 0; row < matrix.getRowSize(); row++) {
			for (int column = 0; column < matrix.getColumnSize(); column++) {
				if (!MathUtility.equal(matrix.getValue(row, column), array.getFloat(row, column))) {
					return false;
				}
			}
		}
		return true;
	}

	protected abstract GradientUpdater<?> getOldFunction(int[] shape);

	protected abstract Learner getNewFunction(int[] shape);

	@Test
	public void testGradient() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int[] shape = { 5, 2 };
			INDArray array = Nd4j.linspace(-2.5D, 2.0D, 10).reshape(shape);
			GradientUpdater<?> oldFunction = getOldFunction(shape);
			DenseMatrix gradient = getMatrix(array);
			Map<String, MathMatrix> gradients = new HashMap<>();
			gradients.put("gradients", gradient);
			Learner newFuction = getNewFunction(shape);
			newFuction.doCache(gradients);

			for (int iteration = 0; iteration < 10; iteration++) {
				oldFunction.applyUpdater(array, iteration, 0);
				newFuction.learn(gradients, iteration, 0);

				System.out.println(array);
				System.out.println(gradients);

				Assert.assertTrue(equalMatrix(gradient, array));
			}
		});
		task.get();
	}

	@Test
	public void testModel() {
		int[] shape = { 5, 2 };
		Learner oldModel = getNewFunction(shape);
		for (ModelCodec codec : ModelCodec.values()) {
			byte[] data = codec.encodeModel(oldModel);
			Learner newModel = (Learner) codec.decodeModel(data);
			Assert.assertThat(newModel, CoreMatchers.equalTo(oldModel));
		}
	}

}
