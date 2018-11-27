package com.jstarcraft.module.neuralnetwork.layer;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({
		// 层测试集
		EmbedLayerTestCase.class,

		FMLayerTestCase.class,

		WeightLayerTestCase.class, })
public class LayerTestSuite {

}
