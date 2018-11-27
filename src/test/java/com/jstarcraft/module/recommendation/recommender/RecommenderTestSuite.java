package com.jstarcraft.module.recommendation.recommender;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

import com.jstarcraft.module.recommendation.recommender.benchmark.BenchmarkTestSuite;
import com.jstarcraft.module.recommendation.recommender.collaborative.CollaborativeTestSuite;
import com.jstarcraft.module.recommendation.recommender.content.ContentTestSuite;
import com.jstarcraft.module.recommendation.recommender.context.ContextTestSuite;
import com.jstarcraft.module.recommendation.recommender.extend.ExtendTestSuite;

@RunWith(Suite.class)
@SuiteClasses({
		// 推荐器测试集
		BenchmarkTestSuite.class,

		CollaborativeTestSuite.class,

		ContentTestSuite.class,

		ContextTestSuite.class,

		ExtendTestSuite.class, })
public class RecommenderTestSuite {

}
