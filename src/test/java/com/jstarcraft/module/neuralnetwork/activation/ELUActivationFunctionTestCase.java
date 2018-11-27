package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;

public class ELUActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationELU(0.5F);
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new ELUActivationFunction(0.5F);
	}

}
