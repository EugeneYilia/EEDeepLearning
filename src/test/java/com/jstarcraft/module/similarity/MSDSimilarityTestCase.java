package com.jstarcraft.module.similarity;

import com.jstarcraft.module.similarity.MSDSimilarity;
import com.jstarcraft.module.similarity.Similarity;

public class MSDSimilarityTestCase extends AbstractSimilarityTestCase {

	@Override
	protected boolean checkCorrelation(float correlation) {
		return correlation >= 0F && correlation < Float.POSITIVE_INFINITY;
	}

	@Override
	protected float getIdentical() {
		return 0F;
	}

	// TODO 注意MSD与MSE相似度是计算两个向量的均方误差,范围是0-正无穷.且if (row == column) value = 0D;
	@Override
	protected Similarity getSimilarity() {
		return new MSDSimilarity();
	}

}
