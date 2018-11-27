package com.jstarcraft.module.neuralnetwork.vertex;

import java.util.concurrent.Future;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.model.ModelCodec;

public abstract class VertexTestCase {

	protected MathMatrix getMatrix(INDArray array) {
		return DenseMatrix.valueOf(array.rows(), array.columns(), (row, column, value, message) -> {
			return array.getFloat(row, column);
		});
	}

	private boolean equalMatrix(MathMatrix matrix, INDArray array) {
		for (int row = 0; row < matrix.getRowSize(); row++) {
			for (int column = 0; column < matrix.getColumnSize(); column++) {
				if (!MathUtility.equal(matrix.getValue(row, column), array.getFloat(row, column))) {
					return false;
				}
			}
		}
		return true;
	}

	protected abstract INDArray getError();

	protected abstract int getSize();

	protected abstract GraphVertex getOldFunction();

	protected abstract Vertex getNewFunction();

	@Test
	public void testPropagate() throws Exception {
		EnvironmentContext context = Nd4j.getAffinityManager().getClass().getSimpleName().equals("CpuAffinityManager") ? EnvironmentContext.CPU : EnvironmentContext.GPU;
		Future<?> task = context.doTask(() -> {
			int size = getSize();

			INDArray[] arrays = new INDArray[size];
			for (int index = 0; index < size; index++) {
				INDArray array = Nd4j.rand(5, 2, index);
				arrays[index] = array;
			}
			GraphVertex oldFunction = getOldFunction();

			KeyValue<MathMatrix, MathMatrix>[] samples = new KeyValue[size];
			for (int index = 0; index < size; index++) {
				INDArray array = arrays[index];
				MathMatrix key = getMatrix(array);
				MathMatrix value = getMatrix(array);
				value.setValues(0F);
				samples[index] = new KeyValue<>(key, value);
			}
			Vertex newFuction = getNewFunction();
			newFuction.doCache(samples);

			// 正向传播
			oldFunction.setInputs(arrays);
			INDArray value = oldFunction.doForward(true);
			newFuction.doForward();
			KeyValue<MathMatrix, MathMatrix> output = newFuction.getOutputKeyValue();
			System.out.println(value);
			System.out.println(output.getKey());
			Assert.assertTrue(equalMatrix(output.getKey(), value));

			// 反向传播
			INDArray epsilon = getError();
			oldFunction.setEpsilon(epsilon);
			INDArray[] epsilons = oldFunction.doBackward(false).getValue();
			output.getValue().mapValues(MatrixMapper.copyOf(getMatrix(epsilon)), null, MathCalculator.PARALLEL);
			newFuction.doBackward();
			for (int index = 0; index < size; index++) {
				INDArray array = epsilons[index];
				MathMatrix error = newFuction.getInputKeyValue(index).getValue();
				System.out.println(array);
				System.out.println(error);
				Assert.assertTrue(equalMatrix(error, array));
			}
		});
		task.get();
	}

	@Test
	public void testModel() {
		Vertex oldModel = getNewFunction();
		for (ModelCodec codec : ModelCodec.values()) {
			byte[] data = codec.encodeModel(oldModel);
			Vertex newModel = (Vertex) codec.decodeModel(data);
			Assert.assertThat(newModel, CoreMatchers.equalTo(oldModel));
		}
	}

}
