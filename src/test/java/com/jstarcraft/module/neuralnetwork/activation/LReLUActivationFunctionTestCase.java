package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;

public class LReLUActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationLReLU();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new LReLUActivationFunction();
	}

}
