/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.CompositeMatrix;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

/**
 * A layer with parameters
 * 
 * @author Adam Gibson
 */
@ModelDefinition(value = { "numberOfInputs", "numberOfOutputs", "configurators", "parameters", "gradients", "mode", "function" })
public abstract class AbstractLayer implements Layer {

	protected static MathMatrix getMatrix(MathMatrix matrix) {
		if (matrix instanceof CompositeMatrix) {
			CompositeMatrix matrixes = CompositeMatrix.class.cast(matrix);
			if (matrixes.getComponentSize() == 1) {
				return matrixes.getComponentMatrix(0);
			}
		}
		return matrix;
	}

	protected int numberOfInputs, numberOfOutputs;

	/** 键为inputData(不可以为null),值为outerError(可以为null) */
	protected KeyValue<MathMatrix, MathMatrix> inputKeyValue;

	/** 键为outputData(不可以为null),值为innerError(不可以为null) */
	protected KeyValue<MathMatrix, MathMatrix> outputKeyValue;

	/** 键为middleData(不可以为null),值为middleError(不可以为null) */
	protected KeyValue<MathMatrix, MathMatrix> middleKeyValue;

	protected Map<String, ParameterConfigurator> configurators;

	/** 参数与梯度 */
	protected Map<String, MathMatrix> parameters, gradients;

	protected Mode mode;

	protected ActivationFunction function;

	protected AbstractLayer() {
	}

	protected AbstractLayer(int numberOfInputs, int numberOfOutputs, Map<String, ParameterConfigurator> configurators, Mode mode, ActivationFunction function) {
		this.numberOfInputs = numberOfInputs;
		this.numberOfOutputs = numberOfOutputs;
		this.mode = mode;
		this.function = function;
		this.configurators = configurators;
		this.parameters = new HashMap<>();
		this.gradients = new HashMap<>();
	}

	@Override
	public void doCache(MatrixFactory factory, KeyValue<MathMatrix, MathMatrix> samples) {
		inputKeyValue = samples;
		int rowSize = inputKeyValue.getKey().getRowSize();
		int columnSize = inputKeyValue.getKey().getColumnSize();

		// 检查维度
		if (columnSize != numberOfInputs) {
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
	public KeyValue<MathMatrix, MathMatrix> getInputKeyValue() {
		return inputKeyValue;
	}

	@Override
	public KeyValue<MathMatrix, MathMatrix> getMiddleKeyValue() {
		return middleKeyValue;
	}

	@Override
	public KeyValue<MathMatrix, MathMatrix> getOutputKeyValue() {
		return outputKeyValue;
	}

	@Override
	public void regularize() {
		for (Entry<String, ParameterConfigurator> term : configurators.entrySet()) {
			ParameterConfigurator configurator = term.getValue();
			float l1Regularization = configurator.getL1Regularization();
			float l2Regularization = configurator.getL2Regularization();
			MathMatrix parameter = parameters.get(term.getKey());
			MathMatrix gradient = gradients.get(term.getKey());

			if (l2Regularization > 0D && parameter != null && gradient != null) {
				// TODO 此处可以优化性能
				gradient.mapValues((row, column, value, message) -> {
					value = value + (parameter.getValue(row, column) * l2Regularization);
					return value;
				}, null, MathCalculator.SERIAL);
			}
			if (l1Regularization > 0D && parameter != null && gradient != null) {
				// TODO 此处可以优化性能
				gradient.mapValues((row, column, value, message) -> {
					value = value + (FastMath.signum(parameter.getValue(row, column)) * l1Regularization);
					return value;
				}, null, MathCalculator.SERIAL);
			}
		}
	}

	@Override
	public Map<String, MathMatrix> getParameters() {
		return parameters;
	}

	@Override
	public Map<String, MathMatrix> getGradients() {
		return gradients;
	}

	@Override
	public void setMode(Mode mode) {
		this.mode = mode;
	}

	@Override
	public Mode getMode() {
		return mode;
	}

	@Override
	public ActivationFunction getFunction() {
		return function;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			AbstractLayer that = (AbstractLayer) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.numberOfInputs, that.numberOfInputs);
			equal.append(this.numberOfOutputs, that.numberOfOutputs);
			equal.append(this.configurators, that.configurators);
			equal.append(this.parameters, that.parameters);
			equal.append(this.mode, that.mode);
			equal.append(this.function, that.function);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(numberOfInputs);
		hash.append(numberOfOutputs);
		hash.append(configurators);
		hash.append(parameters);
		hash.append(mode);
		hash.append(function);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return getClass().getSimpleName() + "(numberOfInputs=" + numberOfInputs + ", numberOfOutputs=" + numberOfOutputs + ", configurators=" + configurators + ", parameters=" + parameters + ", mode=" + mode + ", function=" + function + ")";
	}

}
