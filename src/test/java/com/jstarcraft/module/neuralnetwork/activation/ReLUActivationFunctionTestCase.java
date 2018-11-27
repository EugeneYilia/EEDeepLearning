package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;

public class ReLUActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationReLU();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new ReLUActivationFunction();
	}

}
