package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;

public class IdentityActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationIdentity();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new IdentityActivationFunction();
	}

}
