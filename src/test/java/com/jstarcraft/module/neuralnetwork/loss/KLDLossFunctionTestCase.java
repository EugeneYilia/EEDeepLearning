package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossKLD;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class KLDLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossKLD();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new KLDLossFunction();
	}

}
