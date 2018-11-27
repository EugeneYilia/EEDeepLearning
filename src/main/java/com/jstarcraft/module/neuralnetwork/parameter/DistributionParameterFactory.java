package com.jstarcraft.module.neuralnetwork.parameter;

import com.jstarcraft.module.math.algorithm.distribution.Probability;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

public class DistributionParameterFactory implements ParameterFactory {

	private Probability<Number> probability;

	DistributionParameterFactory() {
	}

	public DistributionParameterFactory(Probability<Number> probability) {
		this.probability = probability;
	}

	@Override
	public void setValues(MathMatrix matrix) {
		// TODO Auto-generated method stub
	}

}
