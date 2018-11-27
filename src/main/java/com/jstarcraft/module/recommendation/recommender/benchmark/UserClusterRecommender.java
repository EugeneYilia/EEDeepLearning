package com.jstarcraft.module.recommendation.recommender.benchmark;

import java.util.Map.Entry;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * It is a graphical model that clusters users into K groups for recommendation,
 * see reference: Barbieri et al., <strong>Probabilistic Approaches to
 * Recommendations</strong> (Section 2.2), Synthesis Lectures on Data Mining and
 * Knowledge Discovery, 2014.
 *
 * @author Guo Guibing and Zhang Haidong
 */
@ModelDefinition(value = { "userDimension", "itemDimension", "userTopicProbabilities", "numberOfFactors", "scoreIndexes", "topicScoreMatrix" })
public class UserClusterRecommender extends ProbabilisticGraphicalRecommender {

	/** 用户的每评分次数 */
	private DenseMatrix userScoreMatrix; // Nur
	/** 用户的总评分次数 */
	private DenseVector userScoreVector; // Nu

	/** 主题的每评分概率 */
	private DenseMatrix topicScoreMatrix; // Pkr
	/** 主题的总评分概率 */
	private DenseVector topicScoreVector; // Pi

	/** 用户主题概率映射 */
	private DenseMatrix userTopicProbabilities; // Gamma_(u,k)

	@Override
	protected boolean isConverged(int iter) {
		// TODO 需要重构
		float loss = 0F;

		for (int u = 0; u < numberOfUsers; u++) {
			for (int k = 0; k < numberOfFactors; k++) {
				float ruk = userTopicProbabilities.getValue(u, k);
				float pi_k = topicScoreVector.getValue(k);

				float sum_nl = 0F;
				for (int r = 0; r < scoreIndexes.size(); r++) {
					float nur = userScoreMatrix.getValue(u, r);
					float pkr = topicScoreMatrix.getValue(k, r);

					sum_nl += nur * Math.log(pkr);
				}

				loss += ruk * (Math.log(pi_k) + sum_nl);
			}
		}

		float deltaLoss = (float) (loss - currentLoss);

		if (iter > 1 && (deltaLoss > 0 || Float.isNaN(deltaLoss))) {
			return true;
		}

		currentLoss = loss;
		return false;
	}

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		topicScoreMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfScores);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = topicScoreMatrix.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfScores) + 1;
			});
		}
		topicScoreVector = DenseVector.valueOf(numberOfFactors);
		topicScoreVector.normalize((index, value, message) -> {
			// 防止为0
			return RandomUtility.randomInteger(numberOfFactors) + 1;
		});
		// TODO
		topicScoreMatrix.mapValues((row, column, value, message) -> {
			return (float) Math.log(value);
		}, null, MathCalculator.SERIAL);
		topicScoreVector.mapValues((index, value, message) -> {
			return (float) Math.log(value);
		}, null, MathCalculator.SERIAL);

		userScoreMatrix = DenseMatrix.valueOf(numberOfUsers, numberOfScores);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector scoreVector = trainMatrix.getRowVector(userIndex);
			for (VectorScalar term : scoreVector) {
				float score = term.getValue();
				int scoreIndex = scoreIndexes.get(score);
				userScoreMatrix.shiftValue(userIndex, scoreIndex, 1);
			}
		}
		userScoreVector = DenseVector.valueOf(numberOfUsers, (index, value, message) -> {
			return trainMatrix.getRowVector(index).getElementSize();
		});
		currentLoss = Float.MIN_VALUE;

		userTopicProbabilities = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
	}

	@Override
	protected void eStep() {
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			DenseVector probabilityVector = userTopicProbabilities.getRowVector(userIndex);
			SparseVector scoreVector = trainMatrix.getRowVector(userIndex);
			if (scoreVector.getElementSize() == 0) {
				probabilityVector.copyVector(topicScoreVector);
			} else {
				probabilityVector.normalize((index, value, message) -> {
					float topicProbability = topicScoreVector.getValue(index);
					for (VectorScalar term : scoreVector) {
						int scoreIndex = scoreIndexes.get(term.getValue());
						float scoreProbability = topicScoreMatrix.getValue(index, scoreIndex);
						topicProbability = topicProbability + scoreProbability;
					}
					return topicProbability;
				});
			}
		}
	}

	@Override
	protected void mStep() {
		topicScoreVector.normalize((index, value, message) -> {
			for (int scoreIndex = 0; scoreIndex < numberOfScores; scoreIndex++) {
				float numerator = 0F, denorminator = 0F;
				for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
					float probability = (float) FastMath.exp(userTopicProbabilities.getValue(userIndex, index));
					numerator += probability * userScoreMatrix.getValue(userIndex, scoreIndex);
					denorminator += probability * userScoreVector.getValue(userIndex);
				}
				float probability = (numerator / denorminator);
				topicScoreMatrix.setValue(index, scoreIndex, probability);
			}
			float sumProbability = 0F;
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				float probability = (float) FastMath.exp(userTopicProbabilities.getValue(userIndex, index));
				sumProbability += probability;
			}
			return sumProbability;
		});
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			float topicProbability = userTopicProbabilities.getValue(userIndex, topicIndex);
			float topicValue = 0F;
			for (Entry<Float, Integer> entry : scoreIndexes.entrySet()) {
				float score = entry.getKey();
				float probability = topicScoreMatrix.getValue(topicIndex, entry.getValue());
				topicValue += score * probability;
			}
			value += topicProbability * topicValue;
		}
		return value;
	}

}
