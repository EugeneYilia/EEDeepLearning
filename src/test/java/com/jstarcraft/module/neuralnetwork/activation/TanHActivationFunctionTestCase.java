package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationTanH;

public class TanHActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationTanH();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new TanHActivationFunction();
	}

}
