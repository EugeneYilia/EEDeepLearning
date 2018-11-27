package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossCosineProximity;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class CosineProximityLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossCosineProximity();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new CosineProximityLossFunction();
	}

}
