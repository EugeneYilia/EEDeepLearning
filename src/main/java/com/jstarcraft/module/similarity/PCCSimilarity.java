package com.jstarcraft.module.similarity;

import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * Pearson Correlation Coefficient (PCC)
 *
 * @author zhanghaidong
 */
public class PCCSimilarity extends AbstractSimilarity {

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
	 * Calculate the similarity between thisList and thatList.
	 *
	 * @param leftScores
	 *            this list
	 * @param rightScores
	 *            that list
	 * @return similarity
	 */
	private float getSimilarity(int count, List<KeyValue<Float, Float>> scoreList) {
		// compute similarity
		if (count < 2) {
			return Float.NaN;
		}
		float leftMean = 0F;
		float rightMean = 0F;
		for (KeyValue<Float, Float> term : scoreList) {
			leftMean += term.getKey();
			rightMean += term.getValue();
		}
		leftMean /= count;
		rightMean /= count;
		float sum = 0F, leftPower = 0F, rightPower = 0F;
		for (KeyValue<Float, Float> term : scoreList) {
			float leftDelta = term.getKey() - leftMean;
			float rightDelta = term.getValue() - rightMean;
			sum += leftDelta * rightDelta;
			leftPower += leftDelta * leftDelta;
			rightPower += rightDelta * rightDelta;
		}
		return (float) (sum / (Math.sqrt(leftPower * rightPower)));
	}
}
