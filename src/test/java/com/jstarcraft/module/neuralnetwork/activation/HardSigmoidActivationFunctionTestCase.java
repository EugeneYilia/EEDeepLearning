package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;

public class HardSigmoidActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationHardSigmoid();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new HardSigmoidActivationFunction();
	}

}
