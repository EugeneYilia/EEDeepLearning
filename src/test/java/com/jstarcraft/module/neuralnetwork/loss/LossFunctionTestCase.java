package com.jstarcraft.module.neuralnetwork.loss;

import java.util.LinkedList;
import java.util.concurrent.Future;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.model.ModelCodec;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import com.jstarcraft.module.neuralnetwork.activation.SigmoidActivationFunction;
import com.jstarcraft.module.neuralnetwork.activation.SoftmaxActivationFunction;

public abstract class LossFunctionTestCase {

	protected static DenseMatrix getMatrix(INDArray array) {
		return DenseMatrix.valueOf(array.rows(), array.columns(), (row, column, value, message) -> {
			return array.getFloat(row, column);
		});
	}

	protected static boolean equalMatrix(MathMatrix matrix, INDArray array) {
		for (int row = 0; row < matrix.getRowSize(); row++) {
			for (int column = 0; column < matrix.getColumnSize(); column++) {
				if (Math.abs(matrix.getValue(row, column) - array.getFloat(row, column)) > MathUtility.EPSILON) {
					return false;
				}
			}
		}
		return true;
	}

	protected abstract ILossFunction getOldFunction();

	protected abstract LossFunction getNewFunction(ActivationFunction function);

	@Test
	public void testScore() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			LinkedList<KeyValue<IActivation, ActivationFunction>> activetionList = new LinkedList<>();
			activetionList.add(new KeyValue<>(new ActivationSigmoid(), new SigmoidActivationFunction()));
			activetionList.add(new KeyValue<>(new ActivationSoftmax(), new SoftmaxActivationFunction()));
			for (KeyValue<IActivation, ActivationFunction> keyValue : activetionList) {
				INDArray array = Nd4j.linspace(-2.5D, 2.0D, 10).reshape(5, 2);
				INDArray labels = Nd4j.create(new double[] { 0D, 1D, 0D, 1D, 0D, 1D, 0D, 1D, 0D, 1D }).reshape(5, 2);
				ILossFunction oldFunction = getOldFunction();
				double value = oldFunction.computeScore(labels, array.dup(), keyValue.getKey(), null, false);

				DenseMatrix input = getMatrix(array);
				DenseMatrix output = DenseMatrix.valueOf(input.getRowSize(), input.getColumnSize());
				ActivationFunction function = keyValue.getValue();
				function.forward(input, output);
				LossFunction newFunction = getNewFunction(function);
				newFunction.doCache(getMatrix(labels), output);
				double score = newFunction.computeScore(getMatrix(labels), output, null);

				System.out.println(value);
				System.out.println(score);

				if (Math.abs(value - score) > MathUtility.EPSILON) {
					Assert.fail();
				}
			}
		});
		task.get();
	}

	@Test
	public void testGradient() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			LinkedList<KeyValue<IActivation, ActivationFunction>> activetionList = new LinkedList<>();
			activetionList.add(new KeyValue<>(new ActivationSigmoid(), new SigmoidActivationFunction()));
			activetionList.add(new KeyValue<>(new ActivationSoftmax(), new SoftmaxActivationFunction()));
			for (KeyValue<IActivation, ActivationFunction> keyValue : activetionList) {
				INDArray array = Nd4j.linspace(-2.5D, 2.0D, 10).reshape(5, 2);
				INDArray labels = Nd4j.create(new double[] { 0D, 1D, 0D, 1D, 0D, 1D, 0D, 1D, 0D, 1D }).reshape(5, 2);
				ILossFunction oldFunction = getOldFunction();
				INDArray value = oldFunction.computeGradient(labels, array.dup(), keyValue.getKey(), null);

				DenseMatrix input = getMatrix(array);
				DenseMatrix output = DenseMatrix.valueOf(input.getRowSize(), input.getColumnSize());
				ActivationFunction function = keyValue.getValue();
				function.forward(input, output);
				DenseMatrix gradient = DenseMatrix.valueOf(input.getRowSize(), input.getColumnSize());
				LossFunction newFunction = getNewFunction(function);
				newFunction.doCache(getMatrix(labels), output);
				newFunction.computeGradient(getMatrix(labels), output, null, gradient);
				function.backward(input, gradient, output);
				System.out.println(value);
				System.out.println(output);
				Assert.assertTrue(equalMatrix(output, value));
			}
		});
		task.get();
	}

	@Test
	public void testModel() {
		LossFunction oldModel = getNewFunction(null);
		for (ModelCodec codec : ModelCodec.values()) {
			byte[] data = codec.encodeModel(oldModel);
			LossFunction newModel = (LossFunction) codec.decodeModel(data);
			Assert.assertThat(newModel, CoreMatchers.equalTo(oldModel));
		}
	}

}
