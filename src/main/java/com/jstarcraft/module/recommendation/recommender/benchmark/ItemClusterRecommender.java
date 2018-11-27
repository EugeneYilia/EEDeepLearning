package com.jstarcraft.module.recommendation.recommender.benchmark;

import java.util.Map.Entry;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
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
 * It is a graphical model that clusters items into K groups for recommendation,
 * as opposite to the {@code UserCluster} recommender.
 *
 * @author Guo Guibing and zhanghaidong
 */
@ModelDefinition(value = { "userDimension", "itemDimension", "itemTopicProbabilities", "numberOfFactors", "scoreIndexes", "topicScoreMatrix" })
public class ItemClusterRecommender extends ProbabilisticGraphicalRecommender {

	/** 物品的每评分次数 */
	private DenseMatrix itemScoreMatrix; // Nur
	/** 物品的总评分次数 */
	private DenseVector itemScoreVector; // Nu

	/** 主题的每评分概率 */
	private DenseMatrix topicScoreMatrix; // Pkr
	/** 主题的总评分概率 */
	private DenseVector topicScoreVector; // Pi

	/** 物品主题概率映射 */
	private DenseMatrix itemTopicProbabilities; // Gamma_(u,k)

	@Override
	protected boolean isConverged(int iter) {
		// TODO 需要重构
		float loss = 0F;
		for (int i = 0; i < numberOfItems; i++) {
			for (int k = 0; k < numberOfFactors; k++) {
				float rik = itemTopicProbabilities.getValue(i, k);
				float pi_k = topicScoreVector.getValue(k);

				float sum_nl = 0F;
				for (int scoreIndex = 0; scoreIndex < numberOfScores; scoreIndex++) {
					float nir = itemScoreMatrix.getValue(i, scoreIndex);
					float pkr = topicScoreMatrix.getValue(k, scoreIndex);

					sum_nl += nir * Math.log(pkr);
				}

				loss += rik * (Math.log(pi_k) + sum_nl);
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

		itemScoreMatrix = DenseMatrix.valueOf(numberOfItems, numberOfScores);
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector scoreVector = trainMatrix.getColumnVector(itemIndex);
			for (VectorScalar term : scoreVector) {
				float score = term.getValue();
				int scoreIndex = scoreIndexes.get(score);
				itemScoreMatrix.shiftValue(itemIndex, scoreIndex, 1);
			}
		}
		itemScoreVector = DenseVector.valueOf(numberOfItems, (index, value, message) -> {
			return trainMatrix.getColumnVector(index).getElementSize();
		});
		currentLoss = Float.MIN_VALUE;

		itemTopicProbabilities = DenseMatrix.valueOf(numberOfItems, numberOfFactors);
	}

	@Override
	protected void eStep() {
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			DenseVector probabilityVector = itemTopicProbabilities.getRowVector(itemIndex);
			SparseVector scoreVector = trainMatrix.getColumnVector(itemIndex);
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
				for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
					float probability = (float) FastMath.exp(itemTopicProbabilities.getValue(itemIndex, index));
					numerator += probability * itemScoreMatrix.getValue(itemIndex, scoreIndex);
					denorminator += probability * itemScoreVector.getValue(itemIndex);
				}
				float probability = (numerator / denorminator);
				topicScoreMatrix.setValue(index, scoreIndex, probability);
			}
			float sumProbability = 0F;
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				float probability = (float) FastMath.exp(itemTopicProbabilities.getValue(itemIndex, index));
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
			float topicProbability = itemTopicProbabilities.getValue(itemIndex, topicIndex); // probability
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
