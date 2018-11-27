package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;

public class SoftmaxActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationSoftmax();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new SoftmaxActivationFunction();
	}

}
