package com.jstarcraft.module.similarity;

import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * Calculate Mean Squared Difference (MSD) similarity proposed by Shardanand and
 * Maes [1995]: <i>Social information filtering: Algorithms for automating "word
 * of mouth"</i>
 * <p>
 * Mean Squared Difference (MSD) Similarity
 *
 * @author zhanghaidong
 */
public class MSDSimilarity extends AbstractSimilarity {

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
	
	@Override
	public float getIdentical() {
		return 0F;
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
		if (count == 0) {
			return Float.NaN;
		}
		float sum = 0F;
		for (KeyValue<Float, Float> term : scoreList) {
			sum += Math.pow(term.getKey() - term.getValue(), 2);
		}
		float similarity = count / sum;
		if (Float.isInfinite(similarity)) {
			similarity = 1F;
		}
		return similarity;
	}
}
