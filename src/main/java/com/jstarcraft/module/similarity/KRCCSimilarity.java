package com.jstarcraft.module.similarity;

import java.util.Iterator;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * J. I. Marden, Analyzing and modeling rank data. Boca Raton, Florida: CRC
 * Press, 1996. Mingming Chen etc. A Ranking-oriented Hybrid Approach to
 * QoS-aware Web Service Recommendation. 2015
 * <p>
 * Kendall Rank Correlation Coefficient
 *
 * @author zhanghaidong
 */
public class KRCCSimilarity extends AbstractSimilarity {

	@Override
	public float getCorrelation(MathVector leftVector, MathVector rightVector, float scale) {
		// compute similarity
		List<KeyValue<Float, Float>> scoreList = getScoreList(leftVector, rightVector);
		int count = scoreList.size();
		float similarity = getSimilarity(count, scoreList);
		// shrink to account for vector size
		if (!Double.isNaN(similarity)) {
			if (scale > 0) {
				similarity *= count / (count + scale);
			}
		}
		return similarity;
	}

	/**
	 * Calculate the similarity between thisList and thatList.
	 *
	 * @param leftScores
	 *            this list
	 * @param rightScores
	 *            that list
	 * @return similarity
	 */
	private float getSimilarity(int count, List<KeyValue<Float, Float>> scoreList) {
		if (count < 2) {
			return Float.NaN;
		}
		float sum = 0F;
		Iterator<KeyValue<Float, Float>> iterator = scoreList.iterator();
		KeyValue<Float, Float> previousTerm = iterator.next();
		KeyValue<Float, Float> nextTerm = null;
		while (iterator.hasNext()) {
			nextTerm = iterator.next();
			float leftDelta = previousTerm.getKey() - nextTerm.getKey();
			float rightDelta = previousTerm.getValue() - nextTerm.getValue();
			if (leftDelta * rightDelta < 0F) {
				sum += 1D;
			}
			previousTerm = nextTerm;
		}
		return 1F - 4F * sum / (count * (count - 1));
	}
	
	@Override
	public float getIdentical() {
		return 1F;
	}
}
