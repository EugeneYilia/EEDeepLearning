package com.jstarcraft.module.neuralnetwork.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.RmsPropUpdater;
import org.nd4j.linalg.learning.config.RmsProp;

import com.jstarcraft.module.neuralnetwork.learn.Learner;
import com.jstarcraft.module.neuralnetwork.learn.RmsPropLearner;

public class RmsPropLearnerTestCase extends LearnerTestCase {

	@Override
	protected GradientUpdater<?> getOldFunction(int[] shape) {
		RmsProp configuration = new RmsProp();
		GradientUpdater<?> oldFunction = new RmsPropUpdater(configuration);
		int length = (int) (shape[0] * configuration.stateSize(shape[1]));
		INDArray view = Nd4j.zeros(length);
		oldFunction.setStateViewArray(view, shape, 'c', true);
		return oldFunction;
	}

	@Override
	protected Learner getNewFunction(int[] shape) {
		Learner newFuction = new RmsPropLearner();
		return newFuction;
	}

}
