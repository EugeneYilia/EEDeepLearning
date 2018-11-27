package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftSign;

public class SoftSignActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationSoftSign();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new SoftSignActivationFunction();
	}

}
