/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package com.jstarcraft.module.neuralnetwork.layer;

import java.util.Map;
import java.util.concurrent.CountDownLatch;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.model.ModelCycle;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

/**
 * Embedding layer: feed-forward layer that expects single integers per example
 * as input (class numbers, in range 0 to numClass-1) as input. This input has
 * shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent
 * one-hot representation. Mathematically, EmbeddingLayer is equivalent to using
 * a DenseLayer with a one-hot representation for the input; however, it can be
 * much more efficient with a large number of classes (as a dense layer +
 * one-hot input does a matrix multiply with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is
 * activationFunction(weights.getRow(i) + bias), hence the weight rows can be
 * considered a vector/embedding for each example.
 * 
 * @author Alex Black
 */
public class EmbedLayer extends WeightLayer implements ModelCycle {

	private KeyValue<MathVector, MathVector>[] weightReferences;

	protected EmbedLayer() {
		super();
	}

	public EmbedLayer(int numberOfInputs, int numberOfOutputs, MatrixFactory factory, Map<String, ParameterConfigurator> configurators, Mode mode, ActivationFunction function) {
		super(numberOfInputs, numberOfOutputs, factory, configurators, mode, function);
		this.weightReferences = new KeyValue[numberOfInputs];
	}

	@Override
	public void doCache(MatrixFactory factory, KeyValue<MathMatrix, MathMatrix> samples) {
		inputKeyValue = samples;
		int rowSize = inputKeyValue.getKey().getRowSize();
		int columnSize = inputKeyValue.getKey().getColumnSize();

		// 检查维度
		if (columnSize != 1) {
			throw new IllegalArgumentException();
		}

		middleKeyValue = new KeyValue<>(null, null);
		outputKeyValue = new KeyValue<>(null, null);

		MathMatrix middleData = factory.makeCache(rowSize, numberOfOutputs);
		middleKeyValue.setKey(middleData);
		MathMatrix middleError = factory.makeCache(rowSize, numberOfOutputs);
		middleKeyValue.setValue(middleError);

		MathMatrix outputData = factory.makeCache(rowSize, numberOfOutputs);
		outputKeyValue.setKey(outputData);
		MathMatrix innerError = factory.makeCache(rowSize, numberOfOutputs);
		outputKeyValue.setValue(innerError);
	}

	@Override
	public void doForward() {
		MathMatrix weightParameters = parameters.get(WEIGHT_KEY);
		MathMatrix biasParameters = parameters.get(BIAS_KEY);
		MathMatrix weightGradients = gradients.get(WEIGHT_KEY);

		MathMatrix inputData = getMatrix(inputKeyValue.getKey());
		MathMatrix middleData = getMatrix(middleKeyValue.getKey());
		MathMatrix outputData = getMatrix(outputKeyValue.getKey());

		// inputData.dotProduct(weightParameters, middleData);
		// TODO 考虑并发操作
		middleData.setValues(0F);
		int rowSize = middleData.getRowSize();
		EnvironmentContext context = EnvironmentContext.getContext();
		CountDownLatch latch = new CountDownLatch(rowSize);
		for (int rowIndex = 0; rowIndex < rowSize; rowIndex++) {
			int cursor = rowIndex;
			context.doStructureByAny(cursor, () -> {
				try {
					int index = (int) inputData.getValue(cursor, 0);
					// 索引为负数代表不输出
					if (index >= 0) {
						KeyValue<MathVector, MathVector> keyValue = weightReferences[index];
						if (keyValue == null) {
							keyValue = new KeyValue(weightParameters.getRowVector(index), weightGradients.getRowVector(index));
							weightReferences[index] = keyValue;
						}
						middleData.getRowVector(cursor).copyVector(keyValue.getKey());
						// for (int columnIndex = 0, columnSize =
						// middleData.getColumnSize(); columnIndex < columnSize;
						// columnIndex++) {
						// double value = weightParameters.getValue(index,
						// columnIndex);
						// middleData.setValue(rowIndex, columnIndex, value);
						// }
					}
				} finally {
					latch.countDown();
				}
			});
		}
		try {
			latch.await();
		} catch (Exception exception) {
			throw new RuntimeException(exception);
		}
		if (biasParameters != null) {
			middleData.addRowVector(biasParameters.getRowVector(0));
			// for (int columnIndex = 0, columnSize =
			// middleData.getColumnSize(); columnIndex < columnSize;
			// columnIndex++) {
			// double bias = biasParameters.getValue(0, columnIndex);
			// middleData.getColumnVector(columnIndex).shiftValues(bias);
			// }
		}

		function.forward(middleData, outputData);

		MathMatrix middleError = middleKeyValue.getValue();
		middleError.setValues(0F);

		MathMatrix innerError = outputKeyValue.getValue();
		innerError.setValues(0F);
	}

	@Override
	public void doBackward() {
		MathMatrix weightParameters = parameters.get(WEIGHT_KEY);
		MathMatrix biasParameters = parameters.get(BIAS_KEY);
		MathMatrix weightGradients = gradients.get(WEIGHT_KEY);
		MathMatrix biasGradients = gradients.get(BIAS_KEY);

		MathMatrix innerError = getMatrix(outputKeyValue.getValue());
		MathMatrix middleError = getMatrix(middleKeyValue.getValue());
		// 必须为null
		MathMatrix outerError = getMatrix(inputKeyValue.getValue());
		MathMatrix inputData = getMatrix(inputKeyValue.getKey());
		MathMatrix middleData = getMatrix(middleKeyValue.getKey());
		MathMatrix outputData = getMatrix(outputKeyValue.getKey());

		// 计算梯度
		function.backward(middleData, innerError, middleError);

		// inputData.transposeProductThat(middleError, weightGradients);
		weightGradients.setValues(0F);
		int rowSize = middleData.getRowSize();
		for (int rowIndex = 0; rowIndex < rowSize; rowIndex++) {
			// TODO 此处可以想办法支持并发,得注意根据index同步.
			int index = (int) inputData.getValue(rowIndex, 0);
			if (index >= 0) {
				KeyValue<MathVector, MathVector> keyValue = weightReferences[index];
				keyValue.getValue().addVector(middleError.getRowVector(rowIndex));
			}
		}
		if (biasGradients != null) {
			for (int columnIndex = 0, columnSize = biasGradients.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float bias = middleError.getColumnVector(columnIndex).getSum(false);
				biasGradients.setValue(0, columnIndex, bias);
			}
		}
	}

	@Override
	public void beforeSave() {
	}

	@Override
	public void afterLoad() {
		weightReferences = new KeyValue[numberOfInputs];
	}

}
