package com.jstarcraft.module.neuralnetwork.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationHardTanH;

public class HardTanHActivationFunctionTestCase extends ActivationFunctionTestCase {

	@Override
	protected IActivation getOldFunction() {
		return new ActivationHardTanH();
	}

	@Override
	protected ActivationFunction getNewFunction() {
		return new HardTanHActivationFunction();
	}

}
