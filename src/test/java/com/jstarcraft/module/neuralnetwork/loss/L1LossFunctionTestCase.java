package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossL1;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class L1LossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossL1();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new L1LossFunction();
	}

}
