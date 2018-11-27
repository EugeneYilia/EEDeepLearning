package com.jstarcraft.module.similarity;

import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * Mean Square Error Similarity
 *
 * @author zhanghaidong
 */
public class MSESimilarity extends AbstractSimilarity {

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
		float similarity = 0F;
		for (KeyValue<Float, Float> term : scoreList) {
			float delta = term.getKey() - term.getValue();
			similarity += Math.pow(delta, 2);
		}
		return similarity / count;
	}
}
