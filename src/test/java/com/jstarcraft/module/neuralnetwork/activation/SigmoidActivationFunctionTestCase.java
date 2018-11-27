package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;

public class SigmoidActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationSigmoid();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new SigmoidActivationFunction();
	}

}
