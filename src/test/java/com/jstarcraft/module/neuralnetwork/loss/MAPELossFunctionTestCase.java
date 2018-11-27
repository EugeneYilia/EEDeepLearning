package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMAPE;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class MAPELossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossMAPE();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new MAPELossFunction();
	}

}
