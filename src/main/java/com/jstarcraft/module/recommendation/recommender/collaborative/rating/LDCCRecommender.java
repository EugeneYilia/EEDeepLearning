package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import java.util.Map.Entry;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.Probability;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * @author Guo Guibing and zhanghaidong
 */
public class LDCCRecommender extends ProbabilisticGraphicalRecommender {

	// TODO 重构为稀疏矩阵?
	private Table<Integer, Integer, Integer> userTopics, itemTopics; // Zu, Zv

	private DenseMatrix userTopicTimes, itemTopicTimes; // Nui, Nvj
	private DenseVector userRateTimes, itemRateTimes; // Nv

	private DenseMatrix topicTimes;
	private DenseMatrix topicProbabilities;

	private Probability userProbabilities;
	private Probability itemProbabilities;

	private int[][][] rateTopicTimes;

	private int numberOfUserTopics, numberOfItemTopics;

	private float userAlpha, itemAlpha, ratingBeta;

	private DenseMatrix userTopicProbabilities, itemTopicProbabilities;
	private DenseMatrix userTopicSums, itemTopicSums;
	private float[][][] rateTopicProbabilities, rateTopicSums;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		numberOfStatistics = 0;

		numberOfUserTopics = configuration.getInteger("rec.pgm.number.users", 10);
		numberOfItemTopics = configuration.getInteger("rec.pgm.number.items", 10);

		userAlpha = configuration.getFloat("rec.pgm.user.alpha", 1F / numberOfUserTopics);
		itemAlpha = configuration.getFloat("rec.pgm.item.alpha", 1F / numberOfItemTopics);
		ratingBeta = configuration.getFloat("rec.pgm.rating.beta", 1F / numberOfActions);

		userTopicTimes = DenseMatrix.valueOf(numberOfUsers, numberOfUserTopics);
		itemTopicTimes = DenseMatrix.valueOf(numberOfItems, numberOfItemTopics);
		userRateTimes = DenseVector.valueOf(numberOfUsers);
		itemRateTimes = DenseVector.valueOf(numberOfItems);

		rateTopicTimes = new int[numberOfUserTopics][numberOfItemTopics][numberOfActions];
		topicTimes = DenseMatrix.valueOf(numberOfUserTopics, numberOfItemTopics);
		topicProbabilities = DenseMatrix.valueOf(numberOfUserTopics, numberOfItemTopics);
		userProbabilities = new Probability(numberOfUserTopics, VectorMapper.ZERO);
		itemProbabilities = new Probability(numberOfItemTopics, VectorMapper.ZERO);

		userTopics = HashBasedTable.create();
		itemTopics = HashBasedTable.create();

		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			int rateIndex = scoreIndexes.get(rate);

			int userTopic = RandomUtility.randomInteger(numberOfUserTopics);
			int itemTopic = RandomUtility.randomInteger(numberOfItemTopics);

			userTopicTimes.shiftValue(userIndex, userTopic, 1);
			userRateTimes.shiftValue(userIndex, 1);

			itemTopicTimes.shiftValue(itemIndex, itemTopic, 1);
			itemRateTimes.shiftValue(itemIndex, 1);

			rateTopicTimes[userTopic][itemTopic][rateIndex]++;
			topicTimes.shiftValue(userTopic, itemTopic, 1);

			userTopics.put(userIndex, itemIndex, userTopic);
			itemTopics.put(userIndex, itemIndex, itemTopic);
		}

		// parameters
		userTopicSums = DenseMatrix.valueOf(numberOfUsers, numberOfUserTopics);
		itemTopicSums = DenseMatrix.valueOf(numberOfItems, numberOfItemTopics);
		rateTopicProbabilities = new float[numberOfUserTopics][numberOfItemTopics][numberOfActions];
		rateTopicSums = new float[numberOfUserTopics][numberOfItemTopics][numberOfActions];
	}

	@Override
	protected void eStep() {
		// 缓存概率
		float random = 0F;

		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			// TODO 此处可以重构
			int rateIndex = scoreIndexes.get(rate);
			// TODO 此处可以重构
			// user and item's factors
			int userTopic = userTopics.get(userIndex, itemIndex);
			int itemTopic = itemTopics.get(userIndex, itemIndex);

			// remove this observation
			userTopicTimes.shiftValue(userIndex, userTopic, -1);
			userRateTimes.shiftValue(userIndex, -1);

			itemTopicTimes.shiftValue(itemIndex, itemTopic, -1);
			itemRateTimes.shiftValue(itemIndex, -1);

			rateTopicTimes[userTopic][itemTopic][rateIndex]--;
			topicTimes.shiftValue(userTopic, itemTopic, -1);

			int topicIndex = userTopic;
			// TODO
			// 此处topicProbabilities似乎可以与userProbabilities和itemProbabilities整合.
			// Compute P(i, j)
			// 归一化
			topicProbabilities.mapValues((row, column, value, message) -> {
				// Compute Pmn
				float v1 = (userTopicTimes.getValue(userIndex, row) + userAlpha) / (userRateTimes.getValue(userIndex) + numberOfUserTopics * userAlpha);
				float v2 = (userTopicTimes.getValue(topicIndex, column) + itemAlpha) / (itemRateTimes.getValue(itemIndex) + numberOfItemTopics * itemAlpha);
				float v3 = (rateTopicTimes[row][column][rateIndex] + ratingBeta) / (topicTimes.getValue(row, column) + numberOfActions * ratingBeta);
				value = v1 * v2 * v3;
				return value;
			}, null, MathCalculator.SERIAL);
			// Re-sample user factor
			// 计算概率
			userProbabilities.calculate((index, value, message) -> {
				return topicProbabilities.getRowVector(index).getSum(false);
			});
			userTopic = userProbabilities.random();

			itemProbabilities.calculate((index, value, message) -> {
				return topicProbabilities.getColumnVector(index).getSum(false);
			});
			itemTopic = itemProbabilities.random();

			// Add statistics
			userTopicTimes.shiftValue(userIndex, userTopic, 1);
			userRateTimes.shiftValue(userIndex, 1);

			itemTopicTimes.shiftValue(itemIndex, itemTopic, 1);
			itemRateTimes.shiftValue(itemIndex, 1);

			rateTopicTimes[userTopic][itemTopic][rateIndex]++;
			topicTimes.shiftValue(userTopic, itemTopic, 1);

			userTopics.put(userIndex, itemIndex, userTopic);
			itemTopics.put(userIndex, itemIndex, itemTopic);
		}
	}

	@Override
	protected void mStep() {
		// TODO Auto-generated method stub

	}

	@Override
	protected void readoutParams() {
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			for (int topicIndex = 0; topicIndex < numberOfUserTopics; topicIndex++) {
				userTopicSums.shiftValue(userIndex, topicIndex, (userTopicTimes.getValue(userIndex, topicIndex) + userAlpha) / (userRateTimes.getValue(userIndex) + numberOfUserTopics * userAlpha));
			}
		}

		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			for (int topicIndex = 0; topicIndex < numberOfItemTopics; topicIndex++) {
				itemTopicSums.shiftValue(itemIndex, topicIndex, (itemTopicTimes.getValue(itemIndex, topicIndex) + itemAlpha) / (itemRateTimes.getValue(itemIndex) + numberOfItemTopics * itemAlpha));
			}
		}

		for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
			for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
				for (int rateIndex = 0; rateIndex < numberOfActions; rateIndex++) {
					rateTopicSums[userTopic][itemTopic][rateIndex] += (rateTopicTimes[userTopic][itemTopic][rateIndex] + ratingBeta) / (topicTimes.getValue(userTopic, itemTopic) + numberOfActions * ratingBeta);
				}
			}
		}
		numberOfStatistics++;
	}

	/**
	 * estimate the model parameters
	 */
	@Override
	protected void estimateParams() {
		float scale = 1F / numberOfStatistics;
		// TODO
		// 此处可以重构(整合userTopicProbabilities/userTopicSums和itemTopicProbabilities/itemTopicSums)
		userTopicProbabilities = DenseMatrix.copyOf(userTopicSums, (row, column, value, message) -> {
			return value * scale;
		});
		itemTopicProbabilities = DenseMatrix.copyOf(itemTopicSums, (row, column, value, message) -> {
			return value * scale;
		});

		// TODO 此处可以重构(整合rateTopicProbabilities/rateTopicSums)
		for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
			for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
				for (int rateIndex = 0; rateIndex < numberOfActions; rateIndex++) {
					rateTopicProbabilities[userTopic][itemTopic][rateIndex] = rateTopicSums[userTopic][itemTopic][rateIndex] / numberOfStatistics;
				}
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (Entry<Float, Integer> term : scoreIndexes.entrySet()) {
			float rate = term.getKey();
			int rateIndex = term.getValue();
			float probability = 0F; // P(r|u,v)=\sum_{i,j} P(r|i,j)P(i|u)P(j|v)
			for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
				for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
					probability += rateTopicProbabilities[userTopic][itemTopic][rateIndex] * userTopicProbabilities.getValue(userIndex, userTopic) * itemTopicProbabilities.getValue(itemIndex, itemTopic);
				}
			}
			value += rate * probability;
		}
		return value;
	}

	@Override
	protected boolean isConverged(int iter) {
		// Get the parameters
		estimateParams();
		// Compute the perplexity
		float sum = 0F;
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			sum += perplexity(userIndex, itemIndex, rate);
		}
		float perplexity = (float) Math.exp(sum / numberOfActions);
		float delta = perplexity - currentLoss;
		if (numberOfStatistics > 1 && delta > 0) {
			return true;
		}
		currentLoss = perplexity;
		return false;
	}

	private double perplexity(int user, int item, double rate) {
		int rateIndex = (int) (rate / minimumOfScore - 1);
		// Compute P(r | u, v)
		double probability = 0;
		for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
			for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
				probability += rateTopicProbabilities[userTopic][itemTopic][rateIndex] * userTopicProbabilities.getValue(user, userTopic) * itemTopicProbabilities.getValue(item, itemTopic);
			}
		}
		return -Math.log(probability);
	}

}
