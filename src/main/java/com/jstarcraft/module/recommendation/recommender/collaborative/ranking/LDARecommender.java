package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.ArrayList;
import java.util.List;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.algorithm.Probability;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * Latent Dirichlet Allocation for implicit feedback: Tom Griffiths,
 * <strong>Gibbs sampling in the generative model of Latent Dirichlet
 * Allocation</strong>, 2002. <br>
 * <p>
 * <strong>Remarks:</strong> This implementation of LDA is for implicit
 * feedback, where users are regarded as documents and items as words. To
 * directly apply LDA to explicit ratings, Ian Porteous et al. (AAAI 2008,
 * Section Bi-LDA) mentioned that, one way is to treat items as documents and
 * ratings as words. We did not provide such an LDA implementation for explicit
 * ratings. Instead, we provide recommender {@code URP} as an alternative LDA
 * model for explicit ratings.
 *
 * @author guoguibing and Keqiang Wang
 */
public class LDARecommender extends ProbabilisticGraphicalRecommender {

	/**
	 * entry[k, i]: number of tokens assigned to topic k, given item i.
	 */
	private DenseMatrix topicItemNumbers;

	/**
	 * entry[u, k]: number of tokens assigned to topic k, given user u.
	 */
	private DenseMatrix userTopicNumbers;

	/**
	 * topic assignment as list from the iterator of trainMatrix
	 */
	private List<Integer> topicAssignments;

	/**
	 * entry[u]: number of tokens rated by user u.
	 */
	private DenseVector userTokenNumbers;

	/**
	 * entry[k]: number of tokens assigned to topic t.
	 */
	private DenseVector topicTokenNumbers;

	/**
	 * vector of hyperparameters for alpha and beta
	 */
	private DenseVector alpha, beta;

	/**
	 * cumulative statistics of theta, phi
	 */
	private DenseMatrix userTopicSums, topicItemSums;

	/**
	 * posterior probabilities of parameters
	 */
	private DenseMatrix userTopicProbabilities, topicItemProbabilities;

	/**
	 * setup init member method
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// TODO 此处代码可以消除(使用常量Marker代替或者使用binarize.threshold)
		for (MatrixScalar term : trainMatrix) {
			term.setValue(1F);
		}

		userTopicSums = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
		topicItemSums = DenseMatrix.valueOf(numberOfFactors, numberOfItems);

		// initialize count variables.
		userTopicNumbers = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
		userTokenNumbers = DenseVector.valueOf(numberOfUsers);

		topicItemNumbers = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		topicTokenNumbers = DenseVector.valueOf(numberOfFactors);

		// default value:
		// homas L Griffiths and Mark Steyvers. Finding scientific topics.
		// Proceedings of the National Academy of Sciences, 101(suppl
		// 1):5228–5235, 2004.
		/**
		 * Dirichlet hyper-parameters of user-topic distribution: typical value is 50/K
		 */
		float initAlpha = configuration.getFloat("rec.user.dirichlet.prior", 50F / numberOfFactors);
		/**
		 * Dirichlet hyper-parameters of topic-item distribution, typical value is 0.01
		 */
		float initBeta = configuration.getFloat("rec.topic.dirichlet.prior", 0.01F);
		alpha = DenseVector.valueOf(numberOfFactors, (index, value, message) -> {
			return initAlpha;
		});

		beta = DenseVector.valueOf(numberOfItems, (index, value, message) -> {
			return initBeta;
		});

		// The z_u,i are initialized to values in [0, K-1] to determine the
		// initial state of the Markov chain.
		topicAssignments = new ArrayList<>(trainMatrix.getElementSize());
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			int times = (int) (term.getValue());
			for (int time = 0; time < times; time++) {
				int topicIndex = RandomUtility.randomInteger(numberOfFactors); // 0
																				// ~
				// k-1

				// assign a topic t to pair (u, i)
				topicAssignments.add(topicIndex);
				// number of items of user u assigned to topic t.
				userTopicNumbers.shiftValue(userIndex, topicIndex, 1F);
				// total number of items of user u
				userTokenNumbers.shiftValue(userIndex, 1F);
				// number of instances of item i assigned to topic t
				topicItemNumbers.shiftValue(topicIndex, itemIndex, 1F);
				// total number of words assigned to topic t.
				topicTokenNumbers.shiftValue(topicIndex, 1F);
			}
		}
	}

	@Override
	protected void eStep() {
		float sumAlpha = alpha.getSum(false);
		float sumBeta = beta.getSum(false);

		// Gibbs sampling from full conditional distribution
		int assignmentsIndex = 0;

		Probability sampleProbabilities = new Probability(numberOfFactors, VectorMapper.ZERO);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			int times = (int) (term.getValue());
			for (int time = 0; time < times; time++) {
				int topicIndex = topicAssignments.get(assignmentsIndex); // topic

				userTopicNumbers.shiftValue(userIndex, topicIndex, -1F);
				userTokenNumbers.shiftValue(userIndex, -1F);
				topicItemNumbers.shiftValue(topicIndex, itemIndex, -1F);
				topicTokenNumbers.shiftValue(topicIndex, -1F);

				// 计算概率
				sampleProbabilities.calculate((index, value, message) -> {
					return (userTopicNumbers.getValue(userIndex, index) + alpha.getValue(index)) / (userTokenNumbers.getValue(userIndex) + sumAlpha) * (topicItemNumbers.getValue(index, itemIndex) + beta.getValue(itemIndex)) / (topicTokenNumbers.getValue(index) + sumBeta);
				});

				// scaled sample because of unnormalized p[], randomly sampled a
				// new topic t
				topicIndex = sampleProbabilities.random();

				// add newly estimated z_i to count variables
				userTopicNumbers.shiftValue(userIndex, topicIndex, 1F);
				userTokenNumbers.shiftValue(userIndex, 1F);
				topicItemNumbers.shiftValue(topicIndex, itemIndex, 1F);
				topicTokenNumbers.shiftValue(topicIndex, 1F);

				topicAssignments.set(assignmentsIndex, topicIndex);
				assignmentsIndex++;
			}
		}
	}

	@Override
	protected void mStep() {
		float denominator;
		float value;

		// update alpha vector
		float alphaSum = alpha.getSum(false);
		float alphaDigamma = MathUtility.digamma(alphaSum);
		float alphaValue;
		denominator = 0F;
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			value = userTokenNumbers.getValue(userIndex);
			if (value != 0F) {
				denominator += MathUtility.digamma(value + alphaSum) - alphaDigamma;
			}
		}
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			alphaValue = alpha.getValue(topicIndex);
			alphaDigamma = MathUtility.digamma(alphaValue);
			float numerator = 0F;
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				value = userTopicNumbers.getValue(userIndex, topicIndex);
				if (value != 0F) {
					numerator += MathUtility.digamma(value + alphaValue) - alphaDigamma;
				}
			}
			if (numerator != 0F) {
				alpha.setValue(topicIndex, alphaValue * (numerator / denominator));
			}
		}

		// update beta vector
		float betaSum = beta.getSum(false);
		float betaDigamma = MathUtility.digamma(betaSum);
		float betaValue;
		denominator = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			value = topicTokenNumbers.getValue(topicIndex);
			if (value != 0F) {
				denominator += MathUtility.digamma(value + betaSum) - betaDigamma;
			}
		}
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			betaValue = beta.getValue(itemIndex);
			betaDigamma = MathUtility.digamma(betaValue);
			float numerator = 0F;
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				value = topicItemNumbers.getValue(topicIndex, itemIndex);
				if (value != 0F) {
					numerator += MathUtility.digamma(value + betaValue) - betaDigamma;
				}
			}
			if (numerator != 0F) {
				beta.setValue(itemIndex, betaValue * (numerator / denominator));
			}
		}
	}

	/**
	 * Add to the statistics the values of theta and phi for the current state.
	 */
	@Override
	protected void readoutParams() {
		float sumAlpha = alpha.getSum(false);
		float sumBeta = beta.getSum(false);
		float value;
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				value = (userTopicNumbers.getValue(userIndex, topicIndex) + alpha.getValue(topicIndex)) / (userTokenNumbers.getValue(userIndex) + sumAlpha);
				userTopicSums.shiftValue(userIndex, topicIndex, value);
			}
		}

		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				value = (topicItemNumbers.getValue(topicIndex, itemIndex) + beta.getValue(itemIndex)) / (topicTokenNumbers.getValue(topicIndex) + sumBeta);
				topicItemSums.shiftValue(topicIndex, itemIndex, value);
			}
		}
		numberOfStatistics++;
	}

	@Override
	protected void estimateParams() {
		float scale = 1F / numberOfStatistics;
		userTopicProbabilities = DenseMatrix.copyOf(userTopicSums, (row, column, value, message) -> {
			return value * scale;
		});
		topicItemProbabilities = DenseMatrix.copyOf(topicItemSums, (row, column, value, message) -> {
			return value * scale;
		});
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		DenseVector userVector = userTopicProbabilities.getRowVector(userIndex);
		DenseVector itemVector = topicItemProbabilities.getColumnVector(itemIndex);
		return scalar.dotProduct(userVector, itemVector).getValue();
	}

}
