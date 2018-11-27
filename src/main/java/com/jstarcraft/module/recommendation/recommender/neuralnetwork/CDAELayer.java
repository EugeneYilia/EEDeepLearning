/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package com.jstarcraft.module.recommendation.recommender.neuralnetwork;

import java.util.Map;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import com.jstarcraft.module.neuralnetwork.layer.ParameterConfigurator;
import com.jstarcraft.module.neuralnetwork.layer.WeightLayer;

/**
 * Yao et al., <strong>Collaborative Denoising Auto-Encoders for Top-N
 * Recommender Systems, WSDM 2016.
 *
 * @author Ma Chen
 */
public class CDAELayer extends WeightLayer {

	public final static String USER_KEY = "user";

	private int numberOfUsers;

	public CDAELayer(int numberOfUsers, int numberOfInputs, int numberOfOutputs, MatrixFactory factory, Map<String, ParameterConfigurator> configurators, Mode mode, ActivationFunction function) {
		super(numberOfInputs, numberOfOutputs, factory, configurators, mode, function);

		this.numberOfUsers = numberOfUsers;
		if (!this.configurators.containsKey(USER_KEY)) {
			String message = StringUtility.format("参数{}配置缺失.", USER_KEY);
			throw new IllegalArgumentException(message);
		}

		MathMatrix userParameter = factory.makeCache(numberOfUsers, numberOfOutputs);
		configurators.get(USER_KEY).getFactory().setValues(userParameter);
		this.parameters.put(USER_KEY, userParameter);
		MathMatrix userGradient = factory.makeCache(numberOfUsers, numberOfOutputs);
		this.gradients.put(USER_KEY, userGradient);
	}

	@Override
	public float calculateL1Norm() {
		float l1Sum = super.calculateL1Norm();

		Float userRegularization = configurators.get(USER_KEY).getL1Regularization();
		MathMatrix userParameters = parameters.get(USER_KEY);
		if (userRegularization != null && userParameters != null) {
			if (userParameters instanceof Nd4jMatrix) {
				INDArray array = Nd4jMatrix.class.cast(userParameters).getArray();
				float norm = array.norm1Number().floatValue();
				l1Sum += userRegularization * norm;
			} else {
				float norm = 0F;
				for (MatrixScalar term : userParameters) {
					norm += FastMath.abs(term.getValue());
				}
				l1Sum += userRegularization * norm;
			}
		}

		return l1Sum;
	}

	@Override
	public float calculateL2Norm() {
		float l2Sum = super.calculateL2Norm();

		Float userRegularization = configurators.get(USER_KEY).getL2Regularization();
		MathMatrix userParameters = parameters.get(USER_KEY);
		if (userRegularization != null && userParameters != null) {
			if (userParameters instanceof Nd4jMatrix) {
				INDArray array = Nd4jMatrix.class.cast(userParameters).getArray();
				float norm = array.norm2Number().floatValue();
				l2Sum += 0.5F * userRegularization * norm;
			} else {
				double norm = 0F;
				for (MatrixScalar term : userParameters) {
					norm += term.getValue() * term.getValue();
				}
				l2Sum += 0.5F * userRegularization * norm;
			}
		}

		return l2Sum;
	}

	@Override
	public void doCache(MatrixFactory factory, KeyValue<MathMatrix, MathMatrix> samples) {
		// 检查维度
		if (samples.getKey().getRowSize() != numberOfUsers) {
			throw new IllegalArgumentException();
		}

		super.doCache(factory, samples);
	}

	@Override
	public void doForward() {
		MathMatrix weightParameters = parameters.get(WEIGHT_KEY);
		MathMatrix biasParameters = parameters.get(BIAS_KEY);
		MathMatrix userParameters = parameters.get(USER_KEY);

		MathMatrix inputData = inputKeyValue.getKey();
		MathMatrix middleData = middleKeyValue.getKey();
		MathMatrix outputData = outputKeyValue.getKey();

		middleData.dotProduct(inputData, false, weightParameters, false, MathCalculator.PARALLEL);
		middleData.mapValues((row, column, value, message) -> {
			return value + userParameters.getValue(row, column);
		}, null, MathCalculator.PARALLEL);
		if (biasParameters != null) {
			for (int columnIndex = 0, columnSize = middleData.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float bias = biasParameters.getValue(0, columnIndex);
				middleData.getColumnVector(columnIndex).shiftValues(bias);
			}
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
		MathMatrix userParameters = parameters.get(USER_KEY);
		MathMatrix weightGradients = gradients.get(WEIGHT_KEY);
		MathMatrix biasGradients = gradients.get(BIAS_KEY);
		MathMatrix userGradients = gradients.get(USER_KEY);

		MathMatrix inputData = inputKeyValue.getKey();
		MathMatrix middleData = middleKeyValue.getKey();
		MathMatrix outputData = outputKeyValue.getKey();

		MathMatrix innerError = outputKeyValue.getValue();
		MathMatrix middleError = middleKeyValue.getValue();
		MathMatrix outerError = inputKeyValue.getValue();

		// 计算梯度
		function.backward(middleData, innerError, middleError);
		weightGradients.dotProduct(inputData, true, middleError, false, MathCalculator.PARALLEL);
		userGradients.copyMatrix(middleError, false);
		if (biasGradients != null) {
			for (int columnIndex = 0, columnSize = biasGradients.getColumnSize(); columnIndex < columnSize; columnIndex++) {
				float bias = middleError.getColumnVector(columnIndex).getSum(false);
				biasGradients.setValue(0, columnIndex, bias);
			}
		}

		// weightParameters.doProduct(middleError.transpose()).transpose()
		if (outerError != null) {
			// TODO 使用累计的方式计算
			// TODO 需要锁机制,否则并发计算会导致Bug
			outerError.accumulateProduct(middleError, false, weightParameters, true, MathCalculator.PARALLEL);
		}
	}

	@Override
	public String toString() {
		return "CDAELayer";
	}

}
