package com.jstarcraft.module.neuralnetwork.layer;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelCodec;
import com.jstarcraft.module.neuralnetwork.DenseMatrixFactory;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.vertex.LayerVertex;

public abstract class LayerTestCase {

	protected static DenseMatrix getMatrix(INDArray array) {
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

	protected abstract INDArray getData();

	protected abstract INDArray getError();

	protected abstract List<KeyValue<String, String>> getGradients();

	protected abstract AbstractLayer<?> getOldFunction();

	protected abstract Layer getNewFunction(AbstractLayer<?> layer);

	@Test
	public void testPropagate() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			INDArray array = getData();
			MatrixFactory cache = new DenseMatrixFactory();
			AbstractLayer<?> oldFunction = getOldFunction();
			Layer newFuction = getNewFunction(oldFunction);
			LayerVertex newVertex = new LayerVertex("new", cache, newFuction);
			{
				DenseMatrix key = getMatrix(array);
				DenseMatrix value = DenseMatrix.valueOf(key.getRowSize(), key.getColumnSize());
				KeyValue<MathMatrix, MathMatrix> keyValue = new KeyValue<>(key, value);
				newVertex.doCache(keyValue);
			}

			// 正向传播
			oldFunction.setInput(array);
			INDArray value = oldFunction.activate(true);
			newVertex.doForward();
			KeyValue<MathMatrix, MathMatrix> output = newVertex.getOutputKeyValue();
			System.out.println(value);
			System.out.println(output.getKey());
			Assert.assertTrue(equalMatrix(output.getKey(), value));

			// 反向传播
			INDArray previousEpsilon = getError();
			Pair<Gradient, INDArray> keyValue = oldFunction.backpropGradient(previousEpsilon);
			INDArray nextEpsilon = keyValue.getValue();
			output.getValue().mapValues(MatrixMapper.copyOf(getMatrix(previousEpsilon)), null, MathCalculator.PARALLEL);
			newVertex.doBackward();
			MathMatrix error = newVertex.getInputKeyValue(0).getValue();
			System.out.println(nextEpsilon);
			System.out.println(error);
			if (nextEpsilon != null) {
				Assert.assertTrue(equalMatrix(error, nextEpsilon));
			}

			// 梯度
			Map<String, INDArray> oldGradients = keyValue.getKey().gradientForVariable();
			Map<String, MathMatrix> newGradients = newFuction.getGradients();
			for (KeyValue<String, String> gradient : getGradients()) {
				INDArray oldGradient = oldGradients.get(gradient.getKey());
				MathMatrix newGradient = newGradients.get(gradient.getValue());
				System.out.println(oldGradient);
				System.out.println(newGradient);
				Assert.assertTrue(equalMatrix(newGradient, oldGradient));
			}
		});
		task.get();
	}

	@Test
	public void testModel() {
		AbstractLayer<?> oldFunction = getOldFunction();
		Layer oldModel = getNewFunction(oldFunction);
		for (ModelCodec codec : ModelCodec.values()) {
			byte[] data = codec.encodeModel(oldModel);
			Layer newModel = (Layer) codec.decodeModel(data);
			Assert.assertThat(newModel, CoreMatchers.equalTo(oldModel));
		}
	}

}
