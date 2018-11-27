package com.jstarcraft.module.neuralnetwork.loss;

import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import com.jstarcraft.module.neuralnetwork.activation.SoftmaxActivationFunction;

public class BinaryXENTLossFunctionTestCase extends LossFunctionTestCase {

	@Override
	protected ILossFunction getOldFunction() {
		return new LossBinaryXENT();
	}

	@Override
	protected LossFunction getNewFunction(ActivationFunction function) {
		return new BinaryXENTLossFunction(function instanceof SoftmaxActivationFunction);
	}

}
