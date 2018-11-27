package com.jstarcraft.module.recommendation.recommender.collaborative;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;
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
 * Barbieri et al., <strong>Balancing Prediction and Recommendation Accuracy:
 * Hierarchical Latent Factors for Preference Data</strong>, SDM 2012. <br>
 * <p>
 * <strong>Remarks:</strong> this class implements the BH-free method.
 *
 * @author Guo Guibing and haidong zhang
 */
public abstract class BHFreeRecommender extends ProbabilisticGraphicalRecommender {

	private static class TopicTerm {

		private int userTopic;

		private int itemTopic;

		private int rateIndex;

		private TopicTerm(int userTopic, int itemTopic, int rateIndex) {
			this.userTopic = userTopic;
			this.itemTopic = itemTopic;
			this.rateIndex = rateIndex;
		}

		void update(int userTopic, int itemTopic) {
			this.userTopic = userTopic;
			this.itemTopic = itemTopic;
		}

		public int getUserTopic() {
			return userTopic;
		}

		public int getItemTopic() {
			return itemTopic;
		}

		public int getRateIndex() {
			return rateIndex;
		}

	}

	private Table<Integer, Integer, TopicTerm> topicMatrix;

	private float initGamma, initSigma, initAlpha, initBeta;

	/**
	 * number of user communities
	 */
	protected int numberOfUserTopics; // K

	/**
	 * number of item categories
	 */
	protected int numberOfItemTopics; // L

	/**
	 * evaluation of the user u which have been assigned to the user topic k
	 */
	private DenseMatrix user2TopicNumbers;

	/**
	 * observations for the user
	 */
	private DenseVector userNumbers;

	/**
	 * observations associated with community k
	 */
	private DenseVector userTopicNumbers;

	/**
	 * number of user communities * number of topics
	 */
	private DenseMatrix userTopic2ItemTopicNumbers; // Nkl

	/**
	 * number of user communities * number of topics * number of ratings
	 */
	private int[][][] userTopic2ItemTopicRateNumbers, userTopic2ItemTopicItemNumbers; // Nklr,
	// Nkli;

	// parameters
	protected DenseMatrix user2TopicProbabilities, userTopic2ItemTopicProbabilities;
	protected DenseMatrix user2TopicSums, userTopic2ItemTopicSums;
	protected double[][][] userTopic2ItemTopicRateProbabilities, userTopic2ItemTopicItemProbabilities;
	protected double[][][] userTopic2ItemTopicRateSums, userTopic2ItemTopicItemSums;

	private DenseMatrix topicProbabilities;
	private Probability userProbabilities;
	private Probability itemProbabilities;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		numberOfUserTopics = configuration.getInteger("rec.bhfree.user.topic.number", 10);
		numberOfItemTopics = configuration.getInteger("rec.bhfree.item.topic.number", 10);
		initAlpha = configuration.getFloat("rec.bhfree.alpha", 1.0f / numberOfUserTopics);
		initBeta = configuration.getFloat("rec.bhfree.beta", 1.0f / numberOfItemTopics);
		initGamma = configuration.getFloat("rec.bhfree.gamma", 1.0f / numberOfScores);
		initSigma = configuration.getFloat("rec.sigma", 1.0f / numberOfItems);
		numberOfScores = scoreIndexes.size();

		// TODO 考虑重构(整合为UserTopic对象)
		user2TopicNumbers = DenseMatrix.valueOf(numberOfUsers, numberOfUserTopics);
		userNumbers = DenseVector.valueOf(numberOfUsers);

		userTopic2ItemTopicNumbers = DenseMatrix.valueOf(numberOfUserTopics, numberOfItemTopics);
		userTopicNumbers = DenseVector.valueOf(numberOfUserTopics);

		userTopic2ItemTopicRateNumbers = new int[numberOfUserTopics][numberOfItemTopics][numberOfScores];
		userTopic2ItemTopicItemNumbers = new int[numberOfUserTopics][numberOfItemTopics][numberOfItems];

		topicMatrix = HashBasedTable.create();

		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			int rateIndex = scoreIndexes.get(rate);
			int userTopic = RandomUtility.randomInteger(numberOfUserTopics); // user's
			// topic
			// k
			int itemTopic = RandomUtility.randomInteger(numberOfItemTopics); // item's
			// topic
			// l

			user2TopicNumbers.shiftValue(userIndex, userTopic, 1F);
			userNumbers.shiftValue(userIndex, 1F);
			userTopic2ItemTopicNumbers.shiftValue(userTopic, itemTopic, 1F);
			userTopicNumbers.shiftValue(userTopic, 1F);
			userTopic2ItemTopicRateNumbers[userTopic][itemTopic][rateIndex]++;
			userTopic2ItemTopicItemNumbers[userTopic][itemTopic][itemIndex]++;
			TopicTerm topic = new TopicTerm(userTopic, itemTopic, rateIndex);
			topicMatrix.put(userIndex, itemIndex, topic);
		}

		// parameters
		// TODO 考虑重构为一个对象
		user2TopicSums = DenseMatrix.valueOf(numberOfUsers, numberOfUserTopics);
		userTopic2ItemTopicSums = DenseMatrix.valueOf(numberOfUserTopics, numberOfItemTopics);
		userTopic2ItemTopicRateSums = new double[numberOfUserTopics][numberOfItemTopics][numberOfScores];
		userTopic2ItemTopicRateProbabilities = new double[numberOfUserTopics][numberOfItemTopics][numberOfScores];
		userTopic2ItemTopicItemSums = new double[numberOfUserTopics][numberOfItemTopics][numberOfItems];
		userTopic2ItemTopicItemProbabilities = new double[numberOfUserTopics][numberOfItemTopics][numberOfItems];

		topicProbabilities = DenseMatrix.valueOf(numberOfUserTopics, numberOfItemTopics);
		userProbabilities = new Probability(numberOfUserTopics, VectorMapper.ZERO);
		itemProbabilities = new Probability(numberOfItemTopics, VectorMapper.ZERO);
	}

	@Override
	protected void eStep() {
		for (Cell<Integer, Integer, TopicTerm> term : topicMatrix.cellSet()) {
			int userIndex = term.getRowKey();
			int itemIndex = term.getColumnKey();
			TopicTerm topicTerm = term.getValue();
			int rateIndex = topicTerm.getRateIndex();
			int userTopic = topicTerm.getUserTopic();
			int itemTopic = topicTerm.getItemTopic();

			user2TopicNumbers.shiftValue(userIndex, userTopic, -1F);
			userNumbers.shiftValue(userIndex, -1F);
			userTopic2ItemTopicNumbers.shiftValue(userTopic, itemTopic, -1F);
			userTopicNumbers.shiftValue(userTopic, -1F);
			userTopic2ItemTopicRateNumbers[userTopic][itemTopic][rateIndex]--;
			userTopic2ItemTopicItemNumbers[userTopic][itemTopic][itemIndex]--;

			// normalization
			int userTopicIndex = userTopic;
			int itemTopicIndex = itemTopic;
			topicProbabilities.mapValues((row, column, value, message) -> {
				value = (user2TopicNumbers.getValue(userIndex, userTopicIndex) + initAlpha) / (userNumbers.getValue(userIndex) + numberOfUserTopics * initAlpha);
				value *= (userTopic2ItemTopicNumbers.getValue(userTopicIndex, itemTopicIndex) + initBeta) / (userTopicNumbers.getValue(userTopicIndex) + numberOfItemTopics * initBeta);
				value *= (userTopic2ItemTopicRateNumbers[userTopicIndex][itemTopicIndex][rateIndex] + initGamma) / (userTopic2ItemTopicNumbers.getValue(userTopicIndex, itemTopicIndex) + numberOfScores * initGamma);
				value *= (userTopic2ItemTopicItemNumbers[userTopicIndex][itemTopicIndex][itemIndex] + initSigma) / (userTopic2ItemTopicNumbers.getValue(userTopicIndex, itemTopicIndex) + numberOfItems * initSigma);
				return value;
			}, null, MathCalculator.SERIAL);

			// 计算概率
			userProbabilities.calculate((index, value, message) -> {
				return topicProbabilities.getRowVector(index).getSum(false);
			});
			userTopic = userProbabilities.random();

			itemProbabilities.calculate((index, value, message) -> {
				return topicProbabilities.getColumnVector(index).getSum(false);
			});
			itemTopic = itemProbabilities.random();

			topicTerm.update(userTopic, itemTopic);
			// add statistic
			user2TopicNumbers.shiftValue(userIndex, userTopic, 1F);
			userNumbers.shiftValue(userIndex, 1F);
			userTopic2ItemTopicNumbers.shiftValue(userTopic, itemTopic, 1F);
			userTopicNumbers.shiftValue(userTopic, 1F);
			userTopic2ItemTopicRateNumbers[userTopic][itemTopic][rateIndex]++;
			userTopic2ItemTopicItemNumbers[userTopic][itemTopic][itemIndex]++;
		}

	}

	@Override
	protected void mStep() {

	}

	@Override
	protected void readoutParams() {
		for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				user2TopicSums.shiftValue(userIndex, userTopic, (user2TopicNumbers.getValue(userIndex, userTopic) + initAlpha) / (userNumbers.getValue(userIndex) + numberOfUserTopics * initAlpha));
			}
			for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
				userTopic2ItemTopicSums.shiftValue(userTopic, itemTopic, (userTopic2ItemTopicNumbers.getValue(userTopic, itemTopic) + initBeta) / (userTopicNumbers.getValue(userTopic) + numberOfItemTopics * initBeta));
				for (int rateIndex = 0; rateIndex < numberOfScores; rateIndex++) {
					userTopic2ItemTopicRateSums[userTopic][itemTopic][rateIndex] += (userTopic2ItemTopicRateNumbers[userTopic][itemTopic][rateIndex] + initGamma) / (userTopic2ItemTopicNumbers.getValue(userTopic, itemTopic) + numberOfScores * initGamma);
				}
				for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
					userTopic2ItemTopicItemSums[userTopic][itemTopic][itemIndex] += (userTopic2ItemTopicItemNumbers[userTopic][itemTopic][itemIndex] + initSigma) / (userTopic2ItemTopicNumbers.getValue(userTopic, itemTopic) + numberOfItems * initSigma);
				}
			}
		}
		numberOfStatistics++;
	}

	@Override
	protected void estimateParams() {
		float scale = 1F / numberOfStatistics;
		user2TopicProbabilities = DenseMatrix.copyOf(user2TopicSums, (row, column, value, message) -> {
			return value * scale;
		});
		userTopic2ItemTopicProbabilities = DenseMatrix.copyOf(userTopic2ItemTopicSums, (row, column, value, message) -> {
			return value * scale;
		});
		for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
			for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
				for (int rateIndex = 0; rateIndex < numberOfScores; rateIndex++) {
					userTopic2ItemTopicRateProbabilities[userTopic][itemTopic][rateIndex] = userTopic2ItemTopicRateSums[userTopic][itemTopic][rateIndex] * scale;
				}
				for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
					userTopic2ItemTopicItemProbabilities[userTopic][itemTopic][itemIndex] = userTopic2ItemTopicItemSums[userTopic][itemTopic][itemIndex] * scale;
				}
			}
		}
	}

}
