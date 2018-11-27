package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossPoisson;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public class PoissonLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossPoisson();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new PoissonLossFunction();
	}

}
