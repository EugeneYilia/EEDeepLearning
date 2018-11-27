package com.jstarcraft.module.similarity;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({
		// 相似度测试集
		BinaryCosineSimilarityTestCase.class,

		CosineSimilarityTestCase.class,

		CPCSimilarityTestCase.class,

		DiceCoefficientSimilarityTestCase.class,

		ExJaccardSimilarityTestCase.class,

		JaccardSimilarityTestCase.class,

		KRCCSimilarityTestCase.class,

		MSDSimilarityTestCase.class,

		MSESimilarityTestCase.class,

		PCCSimilarityTestCase.class })
public class SimilarityTestSuite {

}
