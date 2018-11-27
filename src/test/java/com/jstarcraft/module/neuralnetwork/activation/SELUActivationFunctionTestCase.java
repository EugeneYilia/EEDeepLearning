package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSELU;

public class SELUActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationSELU();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new SELUActivationFunction();
	}

}
