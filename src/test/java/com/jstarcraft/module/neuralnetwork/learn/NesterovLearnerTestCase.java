package com.jstarcraft.module.neuralnetwork.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NesterovsUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;

import com.jstarcraft.module.neuralnetwork.learn.Learner;
import com.jstarcraft.module.neuralnetwork.learn.NesterovLearner;

public class NesterovLearnerTestCase extends LearnerTestCase {

	@Override
	protected GradientUpdater<?> getOldFunction(int[] shape) {
		Nesterovs configuration = new Nesterovs();
		GradientUpdater<?> oldFunction = new NesterovsUpdater(configuration);
		int length = (int) (shape[0] * configuration.stateSize(shape[1]));
		INDArray view = Nd4j.zeros(length);
		oldFunction.setStateViewArray(view, shape, 'c', true);
		return oldFunction;
	}

	@Override
	protected Learner getNewFunction(int[] shape) {
		Learner newFuction = new NesterovLearner();
		return newFuction;
	}

}
