package com.jstarcraft.module.recommendation.recommender.content;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({
		// 推荐器测试集
		EFMTestCase.class,

		HFTTestCase.class,

		TopicMFATTestCase.class,

		TopicMFMTTestCase.class })
public class ContentTestSuite {

}
