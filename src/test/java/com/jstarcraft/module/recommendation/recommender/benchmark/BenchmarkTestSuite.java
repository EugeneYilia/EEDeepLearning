package com.jstarcraft.module.recommendation.recommender.benchmark;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({
		// 推荐器测试集
		ConstantGuessTestCase.class,

		GlobalAverageTestCase.class,

		ItemAverageTestCase.class,

		ItemClusterTestCase.class,

		MostPolularTestCase.class,

		RandomGuessTestCase.class,

		UserAverageTestCase.class,

		UserClusterTestCase.class })
public class BenchmarkTestSuite {

}
