package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMSLE;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class MSLELossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossMSLE();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new MSLELossFunction();
	}

}
