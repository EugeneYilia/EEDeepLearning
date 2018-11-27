package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftPlus;

public class SoftPlusActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationSoftPlus();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new SoftPlusActivationFunction();
	}

}
