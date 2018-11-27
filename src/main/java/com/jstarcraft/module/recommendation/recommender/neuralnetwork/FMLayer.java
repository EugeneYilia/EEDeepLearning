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

package com.jstarcraft.module.recommendation.recommender.neuralnetwork;

import java.util.Map;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import com.jstarcraft.module.neuralnetwork.layer.ParameterConfigurator;
import com.jstarcraft.module.neuralnetwork.layer.WeightLayer;

public class FMLayer extends WeightLayer {

	private int[] dimensionSizes;

	protected FMLayer() {
		super();
	}

	public FMLayer(int[] dimensionSizes, int numberOfInputs, int numberOfOutputs, MatrixFactory factory, Map<String, ParameterConfigurator> configurators, Mode mode, ActivationFunction function) {
		super(numberOfInputs, numberOfOutputs, factory, configurators, mode, function);
		this.dimensionSizes = dimensionSizes;
	}

	@Override
	public void doCache(MatrixFactory factory, KeyValue<MathMatrix, MathMatrix> samples) {
		inputKeyValue = samples;
		int rowSize = inputKeyValue.getKey().getRowSize();
		int columnSize = inputKeyValue.getKey().getColumnSize();

		// 检查维度
		if (this.dimensionSizes.length != columnSize) {
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

		MathMatrix inputData = inputKeyValue.getKey();
		MathMatrix middleData = middleKeyValue.getKey();
		MathMatrix outputData = outputKeyValue.getKey();

		// inputData.dotProduct(weightParameters, middleData);
		for (int row = 0; row < inputData.getRowSize(); row++) {
			for (int column = 0; column < weightParameters.getColumnSize(); column++) {
				float value = 0F;
				int cursor = 0;
				for (int index = 0; index < inputData.getColumnSize(); index++) {
					value += weightParameters.getValue(cursor + (int) inputData.getValue(row, index), column);
					cursor += dimensionSizes[index];
				}
				middleData.setValue(row, column, value);
			}
		}
		if (biasParameters != null) {
			for (int columnIndex = 0, columnSize = middleData.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float bias = biasParameters.getValue(0, columnIndex);
				middleData.getColumnVector(columnIndex).shiftValues(bias);
			}
		}

		function.forward(middleData, outputData);

		MathMatrix middleError = middleKeyValue.getValue();
		middleError.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);

		MathMatrix innerError = outputKeyValue.getValue();
		innerError.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
	}

	@Override
	public void doBackward() {
		MathMatrix weightParameters = parameters.get(WEIGHT_KEY);
		MathMatrix biasParameters = parameters.get(BIAS_KEY);
		MathMatrix weightGradients = gradients.get(WEIGHT_KEY);
		MathMatrix biasGradients = gradients.get(BIAS_KEY);

		MathMatrix innerError = outputKeyValue.getValue();
		MathMatrix middleError = middleKeyValue.getValue();
		// 必须为null
		MathMatrix outerError = inputKeyValue.getValue();

		MathMatrix inputData = inputKeyValue.getKey();
		MathMatrix middleData = middleKeyValue.getKey();
		MathMatrix outputData = outputKeyValue.getKey();

		// 计算梯度
		function.backward(middleData, innerError, middleError);

		// inputData.transposeProductThat(middleError, weightGradients);
		weightGradients.mapValues(MatrixMapper.ZERO, null, MathCalculator.SERIAL);
		for (int index = 0; index < inputData.getRowSize(); index++) {
			for (int column = 0; column < middleError.getColumnSize(); column++) {
				int cursor = 0;
				for (int dimension = 0; dimension < dimensionSizes.length; dimension++) {
					int point = cursor + (int) inputData.getValue(index, dimension);
					weightGradients.shiftValue(point, column, middleError.getValue(index, column));
					cursor += dimensionSizes[dimension];
				}
			}
		}
		if (biasGradients != null) {
			for (int columnIndex = 0, columnSize = biasGradients.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float bias = middleError.getColumnVector(columnIndex).getSum(false);
				biasGradients.setValue(0, columnIndex, bias);
			}
		}
	}

}
