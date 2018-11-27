package com.jstarcraft.module.neuralnetwork.learn;

import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NoOpUpdater;
import org.nd4j.linalg.learning.config.NoOp;

import com.jstarcraft.module.neuralnetwork.learn.Learner;
import com.jstarcraft.module.neuralnetwork.learn.IgnoreLearner;

public class IgnoreLearnerTestCase extends LearnerTestCase {

	@Override
	protected GradientUpdater<?> getOldFunction(int[] shape) {
		NoOp configuration = new NoOp();
		GradientUpdater<?> oldFunction = new NoOpUpdater(configuration);
		return oldFunction;
	}

	@Override
	protected Learner getNewFunction(int[] shape) {
		Learner newFuction = new IgnoreLearner();
		return newFuction;
	}

}
