package com.jstarcraft.module.recommendation.recommender.extend;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({
		// 推荐器测试集
		AssociationRuleTestCase.class,

		PersonalityDiagnosisTestCase.class,

		PRankDTestCase.class,

		SlopeOneTestCase.class })
public class ExtendTestSuite {

}
