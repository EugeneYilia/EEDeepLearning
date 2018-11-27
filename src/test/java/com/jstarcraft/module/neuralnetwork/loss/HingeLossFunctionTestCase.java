package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossHinge;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class HingeLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossHinge();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new HingeLossFunction();
	}

}
