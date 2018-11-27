package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import com.jstarcraft.module.neuralnetwork.activation.SoftmaxActivationFunction;

public class MCXENTLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossMCXENT();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new MCXENTLossFunction(function instanceof SoftmaxActivationFunction);
	}

}
