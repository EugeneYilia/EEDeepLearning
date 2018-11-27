package com.jstarcraft.module.similarity;

import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * Cosine similarity
 *
 * @author zhanghaidong
 */
public class CosineSimilarity extends AbstractSimilarity {

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
		return 1F;
	}

	/**
	 * calculate the similarity between thisList and thatList.
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
		float power = 0F, leftPower = 0F, rightPower = 0F;
		for (KeyValue<Float, Float> term : scoreList) {
			float leftScore = term.getKey();
			float rightScore = term.getValue();
			power += leftScore * rightScore;
			leftPower += leftScore * leftScore;
			rightPower += rightScore * rightScore;
		}
		return (float) (power / Math.sqrt(leftPower * rightPower));
	}
}
