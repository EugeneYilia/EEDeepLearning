package com.jstarcraft.module.neuralnetwork;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunctionTestSuite;
import com.jstarcraft.module.neuralnetwork.layer.LayerTestSuite;
import com.jstarcraft.module.neuralnetwork.learn.LearnerTestSuite;
import com.jstarcraft.module.neuralnetwork.loss.LossFunctionTestSuite;
import com.jstarcraft.module.neuralnetwork.vertex.VertexTestSuite;

@RunWith(Suite.class)
@SuiteClasses({

		GraphTestCase.class,

		ActivationFunctionTestSuite.class,

		LayerTestSuite.class,

		LearnerTestSuite.class,

		LossFunctionTestSuite.class,

		VertexTestSuite.class })
public class NeuralTestSuite {

}
