package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossSquaredHinge;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class SquaredHingeLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossSquaredHinge();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new SquaredHingeLossFunction();
	}

}
