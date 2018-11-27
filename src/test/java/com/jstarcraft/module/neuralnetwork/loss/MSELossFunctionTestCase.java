package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class MSELossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossMSE();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new MSELossFunction();
	}

}
